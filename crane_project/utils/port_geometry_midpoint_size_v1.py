"""OFFLINE decoded-size objective; the original midpoint head stays unchanged.

Operate on continuous original-coordinate points, BEFORE the online identity
branch and full-B fallback. GT axis association comes from existing cyclic
point matching, never per-prediction edge sorting. Python3.8/torch1.8+.
"""
import torch
from torch.nn import functional as F

SIZE_WEIGHT = .025
SIZE_BETA = .1
MIN_LENGTH = 1e-6


def projected_edges(points):
    """Same orthogonal projection as v1, without detached-B replacement."""
    if (points.ndim != 3 or points.shape[1:] != (4, 2) or not len(points) or
            not bool(torch.isfinite(points).all())):
        raise ValueError('Expected nonempty finite original-coordinate four points')
    u, v = points[:, 0]-points[:, 2], points[:, 1]-points[:, 3]
    direction = torch.stack((u[:, 0]+v[:, 1], u[:, 1]-v[:, 0]), -1)
    if bool((direction.norm(dim=1) <= MIN_LENGTH).any()):
        raise ValueError('Degenerate continuous rectangle direction')
    theta = torch.atan2(direction[:, 1], direction[:, 0])
    axis = torch.stack((theta.cos(), theta.sin()), -1)
    normal = torch.stack((-theta.sin(), theta.cos()), -1)
    edges = torch.stack(((u*axis).sum(-1), (v*normal).sum(-1)), -1)
    if not bool(torch.isfinite(edges).all()) or bool((edges <= MIN_LENGTH).any()):
        raise ValueError('Nonpositive continuous projected edge; no dropping or clamping')
    return edges


def size_loss(points_original, targets):
    """Equal two-axis SmoothL1(log(P/G)); labels detached and cyclic-matched."""
    predicted = projected_edges(points_original)
    target = targets['original'].detach()
    if target.shape != points_original.shape:
        raise ValueError('Matched GT/prediction four-point shapes differ')
    # Opposite GT midpoint vectors give associated lengths even for 90deg
    # cyclic matches, squares and equivalent raw w/h-angle representations.
    edges = torch.stack(((target[:, 0]-target[:, 2]).norm(dim=1),
                         (target[:, 1]-target[:, 3]).norm(dim=1)), -1)
    if not bool(torch.isfinite(edges).all()) or bool((edges <= MIN_LENGTH).any()):
        raise ValueError('Invalid matched GT edge')
    residual = predicted.log()-edges.log()
    parts = F.smooth_l1_loss(residual, torch.zeros_like(residual),
                           beta=SIZE_BETA, reduction='none').mean(dim=0)
    return parts.mean(), dict(parts=parts, log_residual=residual,
                              predicted_edges=predicted, target_edges=edges)
