"""Deterministic structured-background proxies for dark-domain diagnosis.

The proxy has two independent structure families.  Both are applied after a
moderate, geometry-preserving darkening step:

* ``industrial_edges`` draws persistent rails, thin edges, glare, and
  text-like strokes outside target boxes.
* ``source_background`` pastes context-matched patches extracted only from
  non-target regions of source training images.

These utilities never alter box coordinates.  Structural interference is
masked out around targets, while the photometric darkening still affects the
whole image as intended.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from crane_project.utils.dark_degradation import apply_dark_degradation


SUPPORTED_STRUCTURED_PROXY_FAMILIES = (
    'industrial_edges',
    'source_background',
)


def _stable_seed(*parts) -> int:
    payload = ':'.join(str(part) for part in parts).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder='little', signed=False)


def _gt_polygon(gt: Dict) -> np.ndarray:
    rect = (
        (float(gt['cx']), float(gt['cy'])),
        (max(float(gt['w']), 1.0), max(float(gt['h']), 1.0)),
        float(gt['angle']),
    )
    return cv2.boxPoints(rect).astype(np.int32)


def target_exclusion_mask(image_shape: Sequence[int],
                          gts: Sequence[Dict],
                          margin_ratio: float = 0.18) -> np.ndarray:
    """Return a uint8 mask covering targets plus a conservative margin."""
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    max_extent = 0.0
    for gt in gts:
        polygon = _gt_polygon(gt)
        cv2.fillPoly(mask, [polygon], 255)
        max_extent = max(max_extent, float(gt['w']), float(gt['h']))
    if max_extent > 0.0 and margin_ratio > 0.0:
        radius = max(2, int(round(max_extent * float(margin_ratio))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def _parse_dota_polygons(path: str) -> List[np.ndarray]:
    polygons = []
    if not os.path.isfile(path):
        return polygons
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                coords = np.asarray(
                    [float(value) for value in parts[:8]],
                    dtype=np.float32).reshape(4, 2)
            except ValueError:
                continue
            polygons.append(np.rint(coords).astype(np.int32))
    return polygons


def _annotation_exclusion_mask(image_shape: Sequence[int],
                               polygons: Sequence[np.ndarray],
                               margin_px: int) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, list(polygons), 255)
    if margin_px > 0 and bool(mask.any()):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(margin_px) + 1, 2 * int(margin_px) + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def _region_sum(integral: np.ndarray, x: int, y: int,
                width: int, height: int) -> float:
    x2, y2 = x + width, y + height
    return float(
        integral[y2, x2] - integral[y, x2]
        - integral[y2, x] + integral[y, x])


def _patch_stats(image: np.ndarray) -> Dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    return dict(
        mean=float(gray.mean()),
        std=float(gray.std()),
        gradient=float(gradient.mean()),
    )


def build_background_patch_library(
        data_root: str,
        max_patches: int = 128,
        seed: int = 0,
        sequence_prefix: str = 'real_',
        min_texture_std: float = 7.0,
        min_gradient: float = 5.0) -> Dict:
    """Extract deterministic non-target patches from source ``train`` only."""
    root = os.path.realpath(data_root)
    image_dir = os.path.realpath(os.path.join(root, 'train', 'images'))
    ann_dir = os.path.realpath(os.path.join(root, 'train', 'annfiles'))
    expected_image_dir = os.path.join(root, 'train', 'images')
    expected_ann_dir = os.path.join(root, 'train', 'annfiles')
    if image_dir != expected_image_dir or ann_dir != expected_ann_dir:
        raise ValueError('Background library must resolve to source train data')
    if not os.path.isdir(image_dir) or not os.path.isdir(ann_dir):
        raise ValueError(
            f'Missing source train directories: {image_dir}, {ann_dir}')
    if max_patches <= 0:
        raise ValueError('max_patches must be positive')

    paths = sorted(
        path for path in glob.glob(os.path.join(image_dir, '*'))
        if os.path.isfile(path)
        and os.path.basename(path).startswith(sequence_prefix)
        and os.path.splitext(path)[1].lower() in (
            '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
    if not paths:
        raise ValueError(
            f'No source train images match prefix {sequence_prefix!r}')

    rng = np.random.default_rng(_stable_seed(seed, 'background_library'))
    order = rng.permutation(len(paths)).tolist()
    patches = []
    donors = []
    attempts_per_image = 10
    for path_index in order:
        image_path = paths[int(path_index)]
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        ann_path = os.path.join(ann_dir, stem + '.txt')
        if not os.path.isfile(ann_path):
            continue
        polygons = _parse_dota_polygons(ann_path)
        margin = max(4, int(round(min(height, width) * 0.015)))
        exclusion = _annotation_exclusion_mask(
            image.shape, polygons, margin_px=margin)
        integral = cv2.integral((exclusion > 0).astype(np.uint8))
        local_rng = np.random.default_rng(
            _stable_seed(seed, stem, 'patches'))

        for _ in range(attempts_per_image):
            short = float(local_rng.uniform(0.06, 0.16)) * min(height, width)
            aspect = float(np.exp(local_rng.uniform(
                math.log(0.55), math.log(2.8))))
            patch_w = int(np.clip(round(short * math.sqrt(aspect)), 24, width))
            patch_h = int(np.clip(round(short / math.sqrt(aspect)), 24, height))
            if patch_w >= width or patch_h >= height:
                continue
            x = int(local_rng.integers(0, width - patch_w + 1))
            y = int(local_rng.integers(0, height - patch_h + 1))
            if _region_sum(integral, x, y, patch_w, patch_h) > 0.0:
                continue
            patch = image[y:y + patch_h, x:x + patch_w].copy()
            stats = _patch_stats(patch)
            if (stats['std'] < min_texture_std
                    or stats['gradient'] < min_gradient):
                continue
            relative_image = os.path.relpath(image_path, root)
            relative_ann = os.path.relpath(ann_path, root)
            record = dict(
                image=patch,
                stats=stats,
                source_image=relative_image,
                source_annotation=relative_ann,
                source_rect=[x, y, patch_w, patch_h],
            )
            patches.append(record)
            donors.append(dict(
                source_image=relative_image,
                source_annotation=relative_ann,
                source_rect=[x, y, patch_w, patch_h],
                stats=stats,
            ))
            if len(patches) >= max_patches:
                break
        if len(patches) >= max_patches:
            break

    if not patches:
        raise RuntimeError('No valid non-target source background patches')
    manifest_payload = json.dumps(donors, sort_keys=True).encode('utf-8')
    manifest = dict(
        split='train',
        image_dir=os.path.relpath(image_dir, root),
        annotation_dir=os.path.relpath(ann_dir, root),
        sequence_prefix=sequence_prefix,
        patch_count=len(patches),
        sha256=hashlib.sha256(manifest_payload).hexdigest(),
        donors=donors,
    )
    return dict(patches=patches, manifest=manifest)


def _sequence_progress(frame: int, start: int, end: int) -> float:
    if end <= start:
        return 0.5
    return float(np.clip((frame - start) / float(end - start), 0.0, 1.0))


def _layout_rng(seed: int, family: str, sequence: str):
    return np.random.default_rng(_stable_seed(seed, family, sequence, 'layout'))


def _slow_position(rng: np.random.Generator, progress: float,
                   index: int) -> Tuple[float, float]:
    base_x = float(rng.uniform(0.10, 0.90))
    base_y = float(rng.uniform(0.10, 0.90))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    speed = float(rng.uniform(0.15, 0.45))
    radius = float(rng.uniform(0.01, 0.035))
    angle = phase + 2.0 * np.pi * speed * progress + index * 0.37
    return (
        float(np.clip(base_x + radius * math.cos(angle), 0.04, 0.96)),
        float(np.clip(base_y + radius * math.sin(angle), 0.04, 0.96)),
    )


def _blend_overlay(image: np.ndarray, overlay: np.ndarray,
                   alpha: np.ndarray, exclusion: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float32)
    alpha[exclusion > 0] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blended = (image.astype(np.float32) * (1.0 - alpha)
               + overlay.astype(np.float32) * alpha)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def apply_industrial_edge_interference(
        image_bgr: np.ndarray,
        gts: Sequence[Dict],
        sequence: str,
        frame: int,
        start: int,
        end: int,
        strength: float,
        seed: int = 0) -> Tuple[np.ndarray, Dict]:
    """Draw persistent industrial structures outside target regions."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return image_bgr.copy(), dict(placements=[], structure_pixels=0)
    height, width = image_bgr.shape[:2]
    exclusion = target_exclusion_mask(image_bgr.shape, gts)
    overlay = image_bgr.copy()
    alpha = np.zeros((height, width), dtype=np.float32)
    progress = _sequence_progress(frame, start, end)
    rng = _layout_rng(seed, 'industrial_edges', sequence)
    placements = []

    count = 3 + int(round(3.0 * strength))
    for index in range(count):
        x_norm, y_norm = _slow_position(rng, progress, index)
        cx, cy = int(x_norm * width), int(y_norm * height)
        kind = index % 4
        value = int(np.clip(145 + 95 * strength + rng.uniform(-20, 20),
                            0, 255))
        color = (value, int(value * 0.96), int(value * 0.82))
        local = np.zeros((height, width), dtype=np.uint8)
        if kind == 0:
            angle = float(rng.uniform(-0.35, 0.35))
            length = int(width * rng.uniform(0.20, 0.48))
            spacing = max(4, int(height * rng.uniform(0.008, 0.025)))
            thickness = max(1, int(round(1 + 2 * strength)))
            dx, dy = int(math.cos(angle) * length / 2), int(
                math.sin(angle) * length / 2)
            for offset in (-spacing, spacing):
                p1 = (cx - dx, cy - dy + offset)
                p2 = (cx + dx, cy + dy + offset)
                cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
                cv2.line(local, p1, p2, 255, thickness + 2, cv2.LINE_AA)
            kind_name = 'rail'
        elif kind == 1:
            size = int(min(height, width) * rng.uniform(0.04, 0.10))
            thickness = max(1, int(round(1 + strength)))
            points = [
                (cx - size, cy), (cx + size, cy),
                (cx, cy - size), (cx, cy + size),
            ]
            cv2.line(overlay, points[0], points[1], color,
                     thickness, cv2.LINE_AA)
            cv2.line(overlay, points[2], points[3], color,
                     thickness, cv2.LINE_AA)
            cv2.line(local, points[0], points[1], 255,
                     thickness + 2, cv2.LINE_AA)
            cv2.line(local, points[2], points[3], 255,
                     thickness + 2, cv2.LINE_AA)
            kind_name = 'text_stroke'
        elif kind == 2:
            axes = (
                max(6, int(width * rng.uniform(0.018, 0.055))),
                max(4, int(height * rng.uniform(0.010, 0.035))),
            )
            cv2.ellipse(overlay, (cx, cy), axes, float(rng.uniform(0, 180)),
                        0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(local, (cx, cy), axes, 0, 0, 360, 255, -1)
            local = cv2.GaussianBlur(local, (0, 0), sigmaX=max(2, axes[1]))
            kind_name = 'glare'
        else:
            length = int(width * rng.uniform(0.12, 0.30))
            angle = float(rng.uniform(-1.2, 1.2))
            dx, dy = int(math.cos(angle) * length / 2), int(
                math.sin(angle) * length / 2)
            thickness = max(1, int(round(1 + 2 * strength)))
            p1, p2 = (cx - dx, cy - dy), (cx + dx, cy + dy)
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.line(local, p1, p2, 255, thickness + 3, cv2.LINE_AA)
            kind_name = 'metal_edge'
        alpha = np.maximum(alpha, local.astype(np.float32) / 255.0)
        placements.append(dict(
            kind=kind_name, center=[cx, cy], normalized=[x_norm, y_norm]))

    alpha *= float(0.30 + 0.45 * strength)
    output = _blend_overlay(image_bgr, overlay, alpha, exclusion)
    return output, dict(
        placements=placements,
        structure_pixels=int(((alpha > 0.02) & (exclusion == 0)).sum()),
        target_exclusion_pixels=int((exclusion > 0).sum()),
    )


def _candidate_destination(image_shape: Sequence[int], patch_shape,
                           x_norm: float, y_norm: float) -> Tuple[int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
    x = int(round(x_norm * width - patch_w / 2.0))
    y = int(round(y_norm * height - patch_h / 2.0))
    return (
        int(np.clip(x, 0, max(width - patch_w, 0))),
        int(np.clip(y, 0, max(height - patch_h, 0))),
    )


def _context_score(patch: Dict, region: np.ndarray) -> float:
    target = _patch_stats(region)
    source = patch['stats']
    return (
        abs(float(source['mean']) - float(target['mean'])) / 64.0
        + abs(float(source['std']) - float(target['std'])) / 48.0
        + abs(float(source['gradient']) - float(target['gradient'])) / 80.0
    )


def _adapt_patch(patch: np.ndarray, region: np.ndarray,
                 strength: float) -> np.ndarray:
    source = patch.astype(np.float32)
    target = region.astype(np.float32)
    source_mean = source.mean(axis=(0, 1), keepdims=True)
    source_std = source.std(axis=(0, 1), keepdims=True)
    target_mean = target.mean(axis=(0, 1), keepdims=True)
    target_std = target.std(axis=(0, 1), keepdims=True)
    scale = np.clip(target_std / np.maximum(source_std, 3.0), 0.45, 1.65)
    adapted = (source - source_mean) * scale + target_mean
    # Retain some donor contrast so the patch remains a genuine distractor.
    adapted = target_mean + (adapted - target_mean) * (1.0 + 0.35 * strength)
    return np.clip(adapted, 0.0, 255.0).astype(np.uint8)


def apply_source_background_interference(
        image_bgr: np.ndarray,
        gts: Sequence[Dict],
        library: Dict,
        sequence: str,
        frame: int,
        start: int,
        end: int,
        strength: float,
        seed: int = 0,
        sequence_state: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """Paste persistent, context-matched source-train background patches."""
    patches = list(library.get('patches', []))
    if not patches:
        raise ValueError('source_background requires a non-empty patch library')
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return image_bgr.copy(), dict(placements=[], pasted_pixels=0)
    height, width = image_bgr.shape[:2]
    exclusion = target_exclusion_mask(image_bgr.shape, gts)
    output = image_bgr.copy()
    progress = _sequence_progress(frame, start, end)
    rng = _layout_rng(seed, 'source_background', sequence)
    placements = []
    pasted_pixels = 0
    count = 1 + int(strength >= 0.55) + int(strength >= 0.75)
    if sequence_state is None:
        sequence_state = {}
    slots = sequence_state.setdefault('source_background_slots', {})

    for index in range(count):
        x_norm, y_norm = _slow_position(rng, progress, index)
        short_side = int(min(height, width) * (
            0.07 + 0.08 * strength + 0.015 * index))
        base_index = int(_stable_seed(
            seed, sequence, 'source_patch', index) % len(patches))
        candidate_indices = [
            (base_index + offset * 17) % len(patches) for offset in range(8)
        ]
        offsets = [
            (0.0, 0.0), (-0.06, 0.0), (0.06, 0.0),
            (0.0, -0.06), (0.0, 0.06),
            (-0.04, -0.04), (0.04, 0.04),
        ]
        slot = slots.get(str(index))
        if slot is None:
            # Select a context-compatible donor and nearby placement once,
            # then keep both fixed while the base position moves smoothly.
            choices = []
            for patch_index in candidate_indices:
                candidate = patches[patch_index]
                candidate_patch = candidate['image']
                candidate_aspect = (
                    candidate_patch.shape[1]
                    / max(float(candidate_patch.shape[0]), 1.0))
                candidate_w = int(np.clip(
                    round(short_side * math.sqrt(candidate_aspect)),
                    20, width))
                candidate_h = int(np.clip(
                    round(short_side / math.sqrt(candidate_aspect)),
                    20, height))
                if candidate_w >= width or candidate_h >= height:
                    continue
                for offset_index, (dx_norm, dy_norm) in enumerate(offsets):
                    x, y = _candidate_destination(
                        image_bgr.shape, (candidate_h, candidate_w),
                        float(np.clip(x_norm + dx_norm, 0.02, 0.98)),
                        float(np.clip(y_norm + dy_norm, 0.02, 0.98)))
                    if bool(exclusion[
                            y:y + candidate_h, x:x + candidate_w].any()):
                        continue
                    region = output[y:y + candidate_h, x:x + candidate_w]
                    choices.append((
                        _context_score(candidate, region), patch_index,
                        offset_index))
            if not choices:
                continue
            context_score, patch_index, offset_index = min(choices)
            slot = dict(
                patch_index=int(patch_index),
                offset_index=int(offset_index),
                initial_context_score=float(context_score))
            slots[str(index)] = slot
        patch_index = int(slot['patch_index'])
        offset_index = int(slot['offset_index'])
        record = patches[patch_index]
        patch = record['image']
        aspect = patch.shape[1] / max(float(patch.shape[0]), 1.0)
        patch_w = int(np.clip(
            round(short_side * math.sqrt(aspect)), 20, width))
        patch_h = int(np.clip(
            round(short_side / math.sqrt(aspect)), 20, height))
        if patch_w >= width or patch_h >= height:
            continue
        dx_norm, dy_norm = offsets[offset_index]
        x, y = _candidate_destination(
            image_bgr.shape, (patch_h, patch_w),
            float(np.clip(x_norm + dx_norm, 0.02, 0.98)),
            float(np.clip(y_norm + dy_norm, 0.02, 0.98)))
        if bool(exclusion[y:y + patch_h, x:x + patch_w].any()):
            continue
        context_score = _context_score(
            record, output[y:y + patch_h, x:x + patch_w])
        resized = cv2.resize(
            record['image'], (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        region = output[y:y + patch_h, x:x + patch_w]
        adapted = _adapt_patch(resized, region, strength)
        yy = np.linspace(-1.0, 1.0, patch_h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, patch_w, dtype=np.float32)[None, :]
        radius = np.sqrt(xx * xx + yy * yy)
        feather = np.clip((1.0 - radius) / 0.28, 0.0, 1.0)
        feather = cv2.GaussianBlur(
            feather, (0, 0),
            sigmaX=max(1.0, min(patch_h, patch_w) * 0.025))
        alpha = feather[..., None] * float(0.42 + 0.38 * strength)
        blended = (region.astype(np.float32) * (1.0 - alpha)
                   + adapted.astype(np.float32) * alpha)
        output[y:y + patch_h, x:x + patch_w] = np.clip(
            blended, 0.0, 255.0).astype(np.uint8)
        pasted_pixels += int((feather > 0.05).sum())
        placements.append(dict(
            source_image=record['source_image'],
            source_annotation=record['source_annotation'],
            source_rect=record['source_rect'],
            destination_rect=[x, y, patch_w, patch_h],
            normalized=[x_norm, y_norm],
            context_score=float(context_score),
            initial_context_score=float(slot['initial_context_score']),
        ))

    return output, dict(
        placements=placements,
        pasted_pixels=int(pasted_pixels),
        target_exclusion_pixels=int((exclusion > 0).sum()),
        library_sha256=library.get('manifest', {}).get('sha256'),
    )


def apply_structured_dark_proxy(
        image_bgr: np.ndarray,
        gts: Sequence[Dict],
        family: str,
        sequence: str,
        frame: int,
        start: int,
        end: int,
        severity: float,
        seed: int = 0,
        dark_family: str = 'photometric',
        temporal_profile: str = 'ramp-plateau',
        background_library: Optional[Dict] = None,
        sequence_state: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """Apply moderate darkening followed by one structured interference."""
    if family not in SUPPORTED_STRUCTURED_PROXY_FAMILIES:
        raise ValueError(
            f'Unsupported structured family {family!r}; expected one of '
            f'{SUPPORTED_STRUCTURED_PROXY_FAMILIES}')
    darkened, dark_meta = apply_dark_degradation(
        image_bgr, family=dark_family, sequence=sequence, frame=frame,
        start=start, end=end, severity=severity, seed=seed,
        profile=temporal_profile)
    strength = float(dark_meta['strength'])
    if family == 'industrial_edges':
        output, structure_meta = apply_industrial_edge_interference(
            darkened, gts, sequence, frame, start, end, strength, seed)
    else:
        if background_library is None:
            raise ValueError(
                'source_background requires background_library')
        output, structure_meta = apply_source_background_interference(
            darkened, gts, background_library, sequence, frame,
            start, end, strength, seed, sequence_state=sequence_state)
    metadata = dict(
        family=family,
        sequence=str(sequence),
        frame=int(frame),
        severity=float(severity),
        strength=strength,
        seed=int(seed),
        geometry_preserving=True,
        target_geometry_modified=False,
        darkening=dark_meta,
        structure=structure_meta,
    )
    return output, metadata
