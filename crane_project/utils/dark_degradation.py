"""Geometry-preserving dark-domain proxy degradations.

This module deliberately exposes *families* instead of a bag of photometric
parameters.  ``photometric`` reproduces the project's existing sRGB-space
augmentation family, while ``dark_isp`` approximates a physically different
low-exposure sensor/ISP path in linear-light space.  Keeping the family label
explicit is required for leave-one-degradation-out proxy validation.

All random parameters are deterministic for ``(seed, family, sequence,
frame)``.  Global illumination parameters vary smoothly along a sequence;
only sensor noise is sampled independently per frame.  Bounding-box geometry
is never changed.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np


SUPPORTED_DARK_FAMILIES = ('photometric', 'dark_isp')
SUPPORTED_TEMPORAL_PROFILES = ('constant', 'ramp-plateau')


def _stable_seed(*parts) -> int:
    payload = ':'.join(str(part) for part in parts).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder='little', signed=False)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def temporal_strength(frame: int,
                      start: int,
                      end: int,
                      severity: float,
                      profile: str = 'ramp-plateau') -> float:
    """Return a temporally coherent degradation strength in ``[0, 1]``.

    ``ramp-plateau`` keeps a long central dark interval and smooth transitions
    at both sides.  A small edge strength is retained so adjacent frames do not
    jump from fully clean to dark.
    """
    if profile not in SUPPORTED_TEMPORAL_PROFILES:
        raise ValueError(
            f'Unsupported temporal profile {profile!r}; expected one of '
            f'{SUPPORTED_TEMPORAL_PROFILES}')
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0 or profile == 'constant' or end <= start:
        return severity

    position = float(np.clip((frame - start) / float(end - start), 0.0, 1.0))
    ramp = 0.22
    if position < ramp:
        envelope = _smoothstep(position / ramp)
    elif position > 1.0 - ramp:
        envelope = _smoothstep((1.0 - position) / ramp)
    else:
        envelope = 1.0
    return severity * (0.15 + 0.85 * envelope)


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4))


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055)


def _sequence_parameters(seed: int, family: str, sequence: str) -> Dict:
    rng = np.random.default_rng(_stable_seed(seed, family, sequence, 'seq'))
    return dict(
        color_axis=float(rng.uniform(-1.0, 1.0)),
        gradient_lr=float(rng.uniform(-1.0, 1.0)),
        gradient_ud=float(rng.uniform(-1.0, 1.0)),
        vignette=float(rng.uniform(0.75, 1.15)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def _spatial_illumination(height: int,
                          width: int,
                          strength: float,
                          params: Dict,
                          vignette_weight: float) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    gradient = (
        params['gradient_lr'] * xx + params['gradient_ud'] * yy) * 0.12
    radius2 = np.clip(xx * xx + yy * yy, 0.0, 2.0) * 0.5
    falloff = 1.0 - strength * (
        vignette_weight * params['vignette'] * radius2 + gradient)
    return np.clip(falloff, 0.25, 1.35).astype(np.float32)


def _photometric_dark(image_bgr: np.ndarray,
                      strength: float,
                      sequence_params: Dict,
                      frame_rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    image = image_bgr.astype(np.float32) / 255.0
    height, width = image.shape[:2]

    exponent = 1.0 + 2.4 * strength
    gain = 1.0 - 0.38 * strength
    contrast = 1.0 - 0.22 * strength
    color_axis = sequence_params['color_axis']
    # BGR gains: a stable sequence-level cool/warm cast.
    channel_gain = np.array([
        1.0 + 0.12 * color_axis * strength,
        1.0,
        1.0 - 0.10 * color_axis * strength,
    ], dtype=np.float32)
    illumination = _spatial_illumination(
        height, width, strength, sequence_params, vignette_weight=0.18)

    degraded = np.power(np.clip(image, 1e-6, 1.0), exponent)
    degraded *= gain * illumination[..., None]
    degraded *= channel_gain.reshape(1, 1, 3)
    mean = degraded.mean(axis=(0, 1), keepdims=True)
    degraded = mean + (degraded - mean) * contrast

    noise_std = 0.002 + 0.010 * strength
    degraded += frame_rng.normal(0.0, noise_std, degraded.shape).astype(
        np.float32)
    degraded = np.clip(degraded, 0.0, 1.0)
    return (degraded * 255.0 + 0.5).astype(np.uint8), dict(
        exponent=float(exponent),
        gain=float(gain),
        contrast=float(contrast),
        noise_std=float(noise_std),
        color_axis=float(color_axis),
    )


def _dark_isp(image_bgr: np.ndarray,
              strength: float,
              sequence_params: Dict,
              frame_rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Approximate low-exposure capture in linear sensor space.

    The implementation intentionally differs from Gamma augmentation: it
    linearizes sRGB, lowers exposure in stops, applies signal-dependent shot
    noise plus read noise, quantizes a 10-bit signal, and maps it back to sRGB.
    This is a lightweight proxy, not a claim of reproducing a specific camera.
    """
    srgb = image_bgr.astype(np.float32) / 255.0
    linear = _srgb_to_linear(srgb).astype(np.float32)
    height, width = linear.shape[:2]

    exposure_stops = 0.4 + 4.2 * strength
    exposure = float(2.0 ** (-exposure_stops))
    color_axis = sequence_params['color_axis']
    sensor_gain = np.array([
        1.0 + 0.08 * color_axis * strength,
        1.0,
        1.0 - 0.07 * color_axis * strength,
    ], dtype=np.float32)
    illumination = _spatial_illumination(
        height, width, strength, sequence_params, vignette_weight=0.35)
    signal = linear * exposure * illumination[..., None]
    signal *= sensor_gain.reshape(1, 1, 3)

    full_well = float(6000.0 - 3500.0 * strength)
    shot_std = np.sqrt(np.clip(signal, 0.0, None) / full_well)
    read_std = float((1.5 + 8.0 * strength) / full_well)
    noise = frame_rng.normal(0.0, 1.0, signal.shape).astype(np.float32)
    signal = signal + noise * shot_std
    signal += frame_rng.normal(0.0, read_std, signal.shape).astype(np.float32)
    black_offset = float(0.0015 * strength)
    signal = np.clip(signal - black_offset, 0.0, 1.0)

    quant_levels = 1023.0
    signal = np.round(signal * quant_levels) / quant_levels
    degraded = _linear_to_srgb(signal)
    return (np.clip(degraded, 0.0, 1.0) * 255.0 + 0.5).astype(
        np.uint8), dict(
            exposure_stops=float(exposure_stops),
            exposure=float(exposure),
            full_well=float(full_well),
            read_std=float(read_std),
            black_offset=float(black_offset),
            color_axis=float(color_axis),
        )


def apply_dark_degradation(image_bgr: np.ndarray,
                           family: str,
                           sequence: str,
                           frame: int,
                           start: int,
                           end: int,
                           severity: float,
                           seed: int = 0,
                           profile: str = 'ramp-plateau') -> Tuple[np.ndarray,
                                                                  Dict]:
    """Apply one deterministic, geometry-preserving dark proxy degradation."""
    if family not in SUPPORTED_DARK_FAMILIES:
        raise ValueError(
            f'Unsupported dark family {family!r}; expected one of '
            f'{SUPPORTED_DARK_FAMILIES}')
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f'image_bgr must have shape [H, W, 3], got {image_bgr.shape}')
    if image_bgr.dtype != np.uint8:
        raise ValueError(f'image_bgr must be uint8, got {image_bgr.dtype}')

    strength = temporal_strength(
        frame, start, end, severity=severity, profile=profile)
    metadata = dict(
        family=family,
        sequence=str(sequence),
        frame=int(frame),
        severity=float(severity),
        strength=float(strength),
        profile=profile,
        seed=int(seed),
        geometry_preserving=True,
    )
    if strength <= 0.0:
        metadata['parameters'] = {}
        return image_bgr.copy(), metadata

    sequence_params = _sequence_parameters(seed, family, sequence)
    frame_rng = np.random.default_rng(
        _stable_seed(seed, family, sequence, frame, 'frame'))
    if family == 'photometric':
        degraded, parameters = _photometric_dark(
            image_bgr, strength, sequence_params, frame_rng)
    else:
        degraded, parameters = _dark_isp(
            image_bgr, strength, sequence_params, frame_rng)
    metadata['parameters'] = parameters
    return degraded, metadata
