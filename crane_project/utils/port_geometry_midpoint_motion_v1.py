"""OFFLINE paired GT-relative edge changes; no online temporal state.

Use continuous original-coordinate v1 projection before identity/fallback.
Transport the second matched GT frame's unsigned axes to the first by nearest
GT direction. Apply that detached correspondence to predictions too; never
sort predicted sides or change existing per-frame point matching.
"""
import math
import torch
from torch.nn import functional as F

from crane_project.utils.port_geometry_midpoint_size_v1 import projected_edges, MIN_LENGTH

MOTION_WEIGHT = .025
MOTION_BETA = .1
AXIS_TIE_TOLERANCE = 1e-6


def axis_correspondence(target_original):
    """GT only, consecutive pairs flattened [prev0,next0,prev1,next1,...].

    This transports unoriented rectangle axes, not physical object-side IDs.
    At a 45-degree tie choose unchanged order and disclose ambiguity. No pair
    is excluded based on angle, aspect ratio, prediction or diagnostic effect.
    """
    target = target_original.detach()
    if (target.ndim != 3 or target.shape[1:] != (4, 2) or
            not len(target) or len(target) % 2 or not bool(torch.isfinite(target).all())):
        raise ValueError('Expected finite nonempty even matched GT four-point frames')
    vectors = torch.stack((target[:,0]-target[:,2], target[:,1]-target[:,3]), 1)
    lengths = vectors.norm(dim=-1)
    if bool((lengths <= MIN_LENGTH).any()):
        raise ValueError('Degenerate matched GT pair axis')
    axes = vectors/lengths[:,:,None]
    previous, current = axes[0::2], axes[1::2]
    dots = torch.bmm(previous, current.transpose(1, 2)).abs()
    same = dots[:,0,0]+dots[:,1,1]
    exchanged = dots[:,0,1]+dots[:,1,0]
    tie = (same-exchanged).abs() <= AXIS_TIE_TOLERANCE
    swap = (exchanged > same) & ~tie
    gt_previous, gt_current = lengths[0::2], lengths[1::2]
    aligned = torch.where(swap[:,None], gt_current[:,[1,0]], gt_current)
    # Clamp applies only to detached cosine diagnostics after float roundoff.
    cosine = torch.where(swap, dots[:,0,1], dots[:,0,0]).clamp(0.,1.)
    return dict(swap=swap, ambiguous=tie, score_margin=(same-exchanged).abs(),
        transported_angle_deg=torch.acos(cosine)*180./math.pi,
        gt_previous=gt_previous, gt_current=aligned)


def motion_loss(points_original, targets):
    """Mean pair/two-axis SmoothL1(delta log P - delta log GT).

    The loss is blind to a constant multiplicative edge bias across a pair.
    It learns annotation-relative change, not small predicted motion itself.
    """
    if points_original.shape != targets['original'].shape:
        raise ValueError('GT/predicted pair four-point shapes differ')
    correspondence = axis_correspondence(targets['original'])
    predicted = projected_edges(points_original)
    previous, current = predicted[0::2], predicted[1::2]
    current = torch.where(correspondence['swap'][:,None], current[:,[1,0]], current)
    pred_increment = current.log()-previous.log()
    gt_increment = correspondence['gt_current'].log()-correspondence['gt_previous'].log()
    residual = pred_increment-gt_increment
    parts = F.smooth_l1_loss(residual, torch.zeros_like(residual),
                           beta=MOTION_BETA, reduction='none').mean(dim=0)
    return parts.mean(), dict(correspondence, parts=parts,
        predicted_log_increment=pred_increment, gt_log_increment=gt_increment,
        log_increment_residual=residual)
