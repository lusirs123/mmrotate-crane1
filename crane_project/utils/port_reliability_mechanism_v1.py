"""Read-only mechanism diagnostics; no fit, detector edits, or TEST access."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from crane_project.utils.port_structure_reliability_v1 import (
    SETTINGS, balanced_response_loss, canonical_boxes, feature_points, quality_loss)

COMPONENTS = ('center', 'size', 'angle')
REFERENCES = np.array([15., .10, 3.])


def ranks(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('Expected finite one-dimensional values')
    order = np.argsort(a, kind='mergesort')
    result = np.empty(len(a), dtype=float)
    start = 0
    while start < len(a):
        end = start+1
        while end < len(a) and a[order[end]] == a[order[start]]:
            end += 1
        result[order[start:end]] = .5*(start+1+end)
        start = end
    return result


def spearman(a, b):
    if len(a) != len(b):
        raise ValueError('Correlation lengths differ')
    if len(a) < 2:
        return None
    x, y = ranks(a), ranks(b)
    x, y = x-x.mean(), y-y.mean()
    norm = float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.clip(x.dot(y)/norm, -1., 1.)) if norm > 0 else None


def summary(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite summary input')
    if not len(a):
        return dict(count=0, mean=None, p10=None, median=None, p90=None)
    return dict(count=len(a), mean=float(a.mean()), p10=float(np.percentile(a, 10)),
                median=float(np.median(a)), p90=float(np.percentile(a, 90)))


def quality_profile(errors, qualities, component, assessed=None, descriptors=None):
    """Continuous quality regression; AUROC here is only error separation."""
    i = COMPONENTS.index(component)
    e, q = np.asarray(errors, dtype=float), np.asarray(qualities, dtype=float)
    if e.shape != q.shape or e.ndim != 2 or e.shape[1] != 3:
        raise ValueError('Expected equal Nx3 error/quality arrays')
    if not np.isfinite(e).all() or not np.isfinite(q).all() or (e < 0).any() or ((q < 0) | (q > 1)).any():
        raise ValueError('Invalid errors/qualities')
    keep = np.ones(len(e), dtype=bool) if assessed is None else np.asarray(assessed, dtype=bool)
    if keep.shape != (len(e),):
        raise ValueError('Assessment mask differs')
    e, q = e[keep, i], q[keep, i]
    good = e < 15 if component == 'center' else e <= REFERENCES[i]
    target = np.exp(-e/REFERENCES[i])
    ng, nb = int(good.sum()), int((~good).sum())
    auc = float((ranks(q)[good].sum()-ng*(ng+1)/2)/(ng*nb)) if ng and nb else None
    correlations = {}
    for name, values in (descriptors or {}).items():
        if len(values) != len(keep):
            raise ValueError('Descriptor length differs')
        correlations[name] = spearman(q, np.asarray(values)[keep])
    return dict(assessed_outputs=len(e), correct_outputs=ng, incorrect_outputs=nb,
        error=summary(e), quality=summary(q), quality_on_correct=summary(q[good]),
        quality_on_incorrect=summary(q[~good]),
        quality_target_mae=float(np.abs(q-target).mean()) if len(e) else None,
        quality_target_rmse=float(np.sqrt(((q-target)**2).mean())) if len(e) else None,
        spearman_quality_vs_negative_error=spearman(q, -e),
        correct_vs_incorrect_auroc=auc, auroc_defined=bool(ng and nb),
        descriptor_correlations=correlations,
        note='Descriptive associations, not calibration, significance, or proof of a shortcut.')


def zero_response_qualities(arm, p3, boxes, scores, meta):
    """Same trained head, zero ONLY its response input, restore flag even on error.

    This is an out-of-distribution input sensitivity check, not a retrained ROI
    ablation. Response prediction/weights and all image/geometry inputs persist.
    """
    if not arm.use_structure or arm.training:
        raise ValueError('Requires the unchanged structure head in eval mode')
    arm.use_structure = False
    try:
        with torch.no_grad():
            return arm(p3, boxes, scores, meta)['qualities'].detach().cpu().tolist()
    finally:
        arm.use_structure = True


def gradient_vector(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).cpu().double()
                        for p, g in zip(parameters, values)])
    if not bool(torch.isfinite(vector).all()):
        raise ValueError('Nonfinite diagnostic gradient')
    return vector


def gradient_relation(first, second):
    a, b = float(first.norm()), float(second.norm())
    return dict(first_norm=a, second_norm=b, second_over_first=b/a if a else None,
        cosine=float(torch.clamp(first.dot(second)/(a*b), -1., 1.)) if a and b else None,
        summed_norm=float((first+second).norm()))


def structure_gradients(arm, p3, boxes, scores, meta, target, mask, genuine_count, response_target, valid):
    """Decompose the EXISTING loss, without backward(), optimizers or updates."""
    if arm.training or not arm.use_structure or p3.requires_grad or p3.grad_fn is not None:
        raise ValueError('Expected eval head with detached frozen P3')
    if any(p.grad is not None for p in arm.parameters()):
        raise ValueError('Expected no accumulated head gradients')
    with torch.enable_grad():
        online = arm(p3, boxes, scores, meta)
        qloss, parts = quality_loss(online['qualities'], target, mask, genuine_count)
        sloss, channel_losses = balanced_response_loss(online['response_logits'], response_target, valid)
        elem = F.smooth_l1_loss(online['qualities'], target.detach(), reduction='none')*mask.detach()
        wg, wp = (.5, .5) if genuine_count else (0., 1.)
        genuine = wg*parts['genuine']
        probe = wp*parts['probe']
        terms = {}
        for i, name in enumerate(COMPONENTS):
            terms[name] = (wg*elem[:genuine_count, i].sum()/mask[:genuine_count].sum().clamp(min=1.)
                           + wp*elem[genuine_count:, i].sum()/mask[genuine_count:].sum().clamp(min=1.))
        if not torch.allclose(sum(terms.values()), qloss, atol=1e-7, rtol=1e-6):
            raise ValueError('Component decomposition differs from the existing quality loss')
        stem = list(arm.stem.parameters())
        qg, sg = gradient_vector(qloss, stem), gradient_vector(sloss, stem)
        gg, pg = gradient_vector(genuine, stem), gradient_vector(probe, stem)
        result = dict(loss_quality=float(qloss.detach()), loss_structure=float(sloss.detach()),
            loss_structure_channels=channel_losses.detach().cpu().tolist(),
            shared_stem_quality_vs_structure=gradient_relation(qg, sg),
            shared_stem_weighted_genuine_vs_weighted_probes=gradient_relation(gg, pg),
            quality_component_losses={k: float(x.detach()) for k, x in terms.items()},
            component_vs_structure={k: gradient_relation(gradient_vector(x, stem), sg) for k, x in terms.items()},
            note='Local diagnostic at fixed weights; cosine does not establish a training-wide causal conflict.')
    if any(p.grad is not None for p in arm.parameters()) or p3.grad is not None:
        raise ValueError('Diagnostic accumulated gradients')
    return result


def angular_difference(first, second):
    return abs((first-second+math.pi/2) % math.pi-math.pi/2)*180/math.pi


def response_evidence(probabilities, valid, axis_model, boxes_model):
    """Offline response evidence on annotations. No GT-dependent online output.

    Line PCA uses excess above its median inside the predicted B square context;
    flat maps remain undefined. Its unsigned orientation is a diagnostic, not a
    new angle estimate or a visibility/quality gate.
    """
    a = np.asarray(probabilities, dtype=float)
    m = np.asarray(valid, dtype=bool)
    axis = np.asarray(axis_model, dtype=float)
    if a.ndim != 3 or a.shape[0] != 2 or m.shape != a.shape[1:] or axis.shape != (2, 2):
        raise ValueError('Response geometry shape differs')
    if not np.isfinite(a).all() or not np.isfinite(axis).all() or not m.any():
        raise ValueError('Invalid response evidence')
    h, w = m.shape
    y, x = np.mgrid[:h, :w]
    points = np.stack((x, y), -1)*SETTINGS['stride']
    mid = axis.mean(axis=0)
    v = axis[1]-axis[0]
    length_sq = float(v.dot(v))
    if length_sq <= 0:
        raise ValueError('Degenerate annotation axis')
    theta = math.atan2(v[1], v[0])
    peaks = []
    for channel in range(2):
        values = np.where(m, a[channel], -np.inf)
        iy, ix = np.unravel_index(values.argmax(), values.shape)
        peaks.append(dict(xy_model_px=points[iy, ix].tolist(), value=float(a[channel, iy, ix]),
                          distance_to_annotation_midpoint_model_px=float(np.linalg.norm(points[iy, ix]-mid))))
    t = ((points-axis[0])*v).sum(-1)/length_sq
    nearest = axis[0]+np.clip(t, 0, 1)[..., None]*v
    sigma = SETTINGS['stride']*SETTINGS['response_sigma_cells']
    targets = np.stack((np.exp(-.5*((points-mid)**2).sum(-1)/sigma**2),
                        np.exp(-.5*((points-nearest)**2).sum(-1)/sigma**2)))
    masses = []
    for channel in range(2):
        positive, negative = targets[channel]*m, (1-targets[channel])*m
        masses.append(dict(target_weighted_response=float((a[channel]*positive).sum()/positive.sum()),
                           background_weighted_response=float((a[channel]*negative).sum()/negative.sum())))
    local = None
    boxes = np.asarray(boxes_model, dtype=float).reshape(-1, 5)
    if len(boxes):
        b = canonical_boxes(torch.tensor(boxes, dtype=torch.float32))[0].numpy()
        side = max(float(b[2])*SETTINGS['context_side_long_multiple'],
                   SETTINGS['stride']*SETTINGS['context_min_side_cells'])
        region = m & (np.abs(points[..., 0]-b[0]) <= side/2) & (np.abs(points[..., 1]-b[1]) <= side/2)
        if region.any():
            weights = np.maximum(a[1][region]-np.median(a[1][region]), 0.)
            p = points[region]
            total = float(weights.sum())
            local = dict(valid_cells=int(region.sum()), excess_mass=total, direction_defined=False,
                         angle_error_to_annotation_axis_deg=None, angle_error_to_B_long_axis_deg=None,
                         eigenvalue_ratio=None, weighted_center_model_px=None)
            if total > 1e-12:
                center = (weights[:, None]*p).sum(0)/total
                offset = p-center
                covariance = (offset*weights[:, None]).T.dot(offset)/total
                eigenvalues, vectors = np.linalg.eigh(covariance)
                local['weighted_center_model_px'] = center.tolist()
                local['eigenvalue_ratio'] = float(eigenvalues[1]/max(eigenvalues[0], 1e-12))
                if eigenvalues[1]-eigenvalues[0] > 1e-9:
                    direction = math.atan2(vectors[1, 1], vectors[0, 1])
                    local.update(direction_defined=True,
                        angle_error_to_annotation_axis_deg=angular_difference(direction, theta),
                        angle_error_to_B_long_axis_deg=angular_difference(direction, float(b[4])))
    return dict(global_peaks=peaks, annotation_weighted_response=masses, B_context_line_moment=local,
                coordinate_role='model pixels; annotation comparison only; no deployment estimate')


def preview(path, image, meta, probabilities, axis_model, boxes_model):
    """Three diagnostic panels; inverse normalization, no geometric image warp."""
    import cv2
    pixels = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
    norm = meta['img_norm_cfg']
    pixels = pixels*np.asarray(norm['std'])+np.asarray(norm['mean'])
    if norm.get('to_rgb', False):
        pixels = pixels[..., ::-1]
    pixels = np.ascontiguousarray(np.clip(pixels, 0, 255).astype(np.uint8))
    axis = np.rint(axis_model).astype(np.int32)
    boxes = np.asarray(boxes_model, dtype=float).reshape(-1, 5)
    panels = []
    for channel in (None, 0, 1):
        panel = pixels.copy()
        if channel is not None:
            # Cell i is at stride*i, not the center of a resized image cell.
            hh, ww = pixels.shape[:2]
            yy, xx = np.mgrid[:hh, :ww]
            heat = cv2.remap(np.asarray(probabilities[channel], dtype=np.float32),
                (xx/SETTINGS['stride']).astype(np.float32), (yy/SETTINGS['stride']).astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            panel = cv2.addWeighted(panel, .6, cv2.applyColorMap(np.uint8(np.clip(heat*255, 0, 255)), cv2.COLORMAP_JET), .4, 0)
        cv2.line(panel, tuple(axis[0]), tuple(axis[1]), (0, 255, 0), 2)
        for box in boxes:
            polygon = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4]*180/math.pi))
            cv2.polylines(panel, [np.rint(polygon).astype(np.int32)], True, (255, 255, 0), 2)
        label = 'B box / annotation axis' if channel is None else ('center response' if channel == 0 else 'line response')
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (20, 20, 20), -1)
        cv2.putText(panel, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        panels.append(panel)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise OSError('Cannot write diagnostic preview')
