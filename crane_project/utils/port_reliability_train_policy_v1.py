"""Offline TRAIN direction qualification; online branch inputs stay unchanged.

Native real axes are qualification evidence, not a replacement angle target.
Webots OBBs are simulation GT; their derived finite lines are structural proxies.
The old evaluation aspect mask and all continuous quality values are preserved.
"""
import math

import numpy as np
import torch

from crane_project.utils.port_structure_reliability_v1 import (
    SETTINGS, canonical_boxes, quality_targets_original)


POLICY = dict(
    version='port_reliability_train_direction_v1',
    real_direction='validated_native_axis_conversion',
    sim_direction='webots_generated_obb_gt_aspect_ge_1p2',
    sim_structure='obb_derived_finite_line_proxy_no_native_axis',
    evaluation_direction='gt_aspect_ge_1p2_unchanged',
    aspect_threshold=1.2, center_reference_px=15., size_reference_relative=.10,
    angle_reference_deg=3., native_tolerance=dict(
        center_px=.15, endpoint_px=.20, length_px=.20, short_px=.20, angle_deg=.20))


def direction_qualification(gt_original, domain, native_axis=None, conversion_k=None):
    """Validate offline source geometry and bind qualification to ONE original GT.

    No visibility inference. A real sample with a missing/inconsistent native
    axis fails closed. Sim needs only its Webots OBB, never a native-axis file.
    Float32 aspect matches the existing quality-target mask at its boundary.
    """
    gt = canonical_boxes(torch.as_tensor(gt_original, dtype=torch.float32).reshape(1, 5))[0]
    g = gt.cpu().numpy().astype(float)
    aspect = float(gt[2]/gt[3])
    evaluation = aspect >= POLICY['aspect_threshold']
    if domain == 'real':
        a = np.asarray(native_axis, dtype=float)
        if (a.shape != (2, 2) or not np.isfinite(a).all()
                or conversion_k is None or not math.isfinite(conversion_k) or conversion_k <= 1):
            raise ValueError('Real TRAIN requires a validated native axis and conversion k')
        length = float(np.linalg.norm(a[1]-a[0]))
        if length <= 0:
            raise ValueError('Degenerate native axis')
        v = .5*g[2]*np.array([math.cos(g[4]), math.sin(g[4])])
        target = np.stack((g[:2]-v, g[:2]+v))
        errors = dict(center_px=float(np.linalg.norm(a.mean(axis=0)-g[:2])),
            endpoint_px=min(float(np.linalg.norm(a-order, axis=1).max())
                            for order in (target, target[::-1])),
            length_px=abs(length-g[2]), short_px=abs(length/conversion_k-g[3]),
            angle_deg=abs((math.atan2(*(a[1]-a[0])[::-1])-g[4]+math.pi/2)
                          % math.pi-math.pi/2)*180/math.pi)
        if any(errors[k] > POLICY['native_tolerance'][k] for k in errors):
            raise ValueError('Native axis/OBB conversion inconsistent')
        train = True
        source, structure = 'native_axis_converted_obb_gt', 'validated_native_axis'
    elif domain == 'sim':
        if native_axis is not None or conversion_k is not None:
            raise ValueError('Webots sim uses OBB GT; do not supply a native axis or real k')
        train = evaluation
        source, structure = 'webots_generated_obb_gt', POLICY['sim_structure']
    else:
        raise ValueError('Only real/sim TRAIN domains are permitted')
    return dict(policy=POLICY['version'], domain=domain, gt_original=g.tolist(),
                gt_aspect_float32=aspect, direction_gt_source=source,
                structure_source=structure, train_angle_eligible=train,
                evaluation_angle_eligible=evaluation)


def train_quality_targets_original(boxes_original, gt_original, qualification):
    """Common offline target hook for BOTH ROI and structure controls.

    Only the angle mask changes. No source/GT/qualification enters the online
    network; caller passes original GT even when the TRAIN image is transformed.
    """
    if (qualification.get('policy') != POLICY['version']
            or qualification.get('domain') not in ('real', 'sim')
            or type(qualification.get('train_angle_eligible')) is not bool):
        raise ValueError('Explicit validated TRAIN direction qualification required')
    current = canonical_boxes(gt_original.detach().reshape(1, 5))
    bound = current.new_tensor(qualification['gt_original']).reshape(1, 5)
    if not torch.allclose(current, bound, atol=1e-5, rtol=1e-6):
        raise ValueError('Qualification belongs to a different original GT')
    if (SETTINGS['center_reference_px'], SETTINGS['size_reference_relative'],
            SETTINGS['angle_reference_deg'], SETTINGS['angle_assessable_gt_aspect']) != (15., .10, 3., 1.2):
        raise ValueError('Existing target references changed')
    target, mask, errors = quality_targets_original(boxes_original, gt_original)
    mask[:, 2] = float(qualification['train_angle_eligible'])
    return target, mask, errors
