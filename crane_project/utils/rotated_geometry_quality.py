"""Smooth Gaussian geometry surrogates for source-only OBB audits.

These functions are deliberately independent of MMRotate and do not select a
detector checkpoint.  An oriented box is represented by ``[cx, cy, w, h,
angle]`` with the angle in radians, matching the labeller's internal format.
The rectangle is approximated by a Gaussian whose axis variances are
``w^2 / 12`` and ``h^2 / 12``.  The resulting symmetric KL and 2-Wasserstein
distances are ranking diagnostics, not replacements for rotated IoU.
"""

from typing import Tuple

import torch


def _validate_boxes(boxes: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(boxes, torch.Tensor):
        raise TypeError('{} must be a torch.Tensor'.format(name))
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError('{} must have shape [N, 5]'.format(name))
    if not torch.is_floating_point(boxes):
        boxes = boxes.float()
    return boxes


def _broadcast_box_pair(
        candidates: torch.Tensor, targets: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    candidates = _validate_boxes(candidates, 'candidates')
    targets = _validate_boxes(targets, 'targets')
    if candidates.shape[0] == 0:
        return candidates, targets[:0]
    if targets.shape[0] == 1:
        targets = targets.expand(candidates.shape[0], -1)
    elif targets.shape[0] != candidates.shape[0]:
        raise ValueError(
            'targets must contain one box or match candidates: {} vs {}'
            .format(candidates.shape[0], targets.shape[0]))
    return candidates, targets


def rotated_box_covariance(
        boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the 2-D Gaussian covariance of each oriented rectangle."""
    boxes = _validate_boxes(boxes, 'boxes')
    if eps <= 0.0:
        raise ValueError('eps must be positive')
    width = boxes[:, 2].abs().clamp_min(eps)
    height = boxes[:, 3].abs().clamp_min(eps)
    angle = boxes[:, 4]
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    # Variance of a uniform interval spanning the rectangle side.
    var_width = width.square() / 12.0
    var_height = height.square() / 12.0
    cov_xx = (cos_angle.square() * var_width
              + sin_angle.square() * var_height + eps)
    cov_yy = (sin_angle.square() * var_width
              + cos_angle.square() * var_height + eps)
    cov_xy = cos_angle * sin_angle * (var_width - var_height)
    return torch.stack((
        torch.stack((cov_xx, cov_xy), dim=-1),
        torch.stack((cov_xy, cov_yy), dim=-1)), dim=-2)


def _gaussian_pair(
        candidates: torch.Tensor, targets: torch.Tensor, eps: float
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates, targets = _broadcast_box_pair(candidates, targets)
    candidate_mean = candidates[:, :2]
    target_mean = targets[:, :2]
    candidate_cov = rotated_box_covariance(candidates, eps=eps)
    target_cov = rotated_box_covariance(targets, eps=eps)
    return candidate_mean, target_mean, candidate_cov, target_cov


def symmetric_gaussian_kl(
        candidates: torch.Tensor, targets: torch.Tensor,
        eps: float = 1e-6) -> torch.Tensor:
    """Return the symmetric KL divergence for candidate/target Gaussian pairs."""
    mean_a, mean_b, cov_a, cov_b = _gaussian_pair(candidates, targets, eps)
    if mean_a.shape[0] == 0:
        return mean_a.new_zeros((0,))
    eye = torch.eye(2, dtype=cov_a.dtype, device=cov_a.device)
    cov_a = cov_a + eps * eye
    cov_b = cov_b + eps * eye
    delta = (mean_b - mean_a).unsqueeze(-1)
    solve_b_a = torch.linalg.solve(cov_b, cov_a)
    solve_a_b = torch.linalg.solve(cov_a, cov_b)
    solve_b_delta = torch.linalg.solve(cov_b, delta).squeeze(-1)
    solve_a_delta = torch.linalg.solve(cov_a, delta).squeeze(-1)
    mean_term_b = (delta.squeeze(-1) * solve_b_delta).sum(dim=-1)
    mean_term_a = (delta.squeeze(-1) * solve_a_delta).sum(dim=-1)
    trace_term = (torch.diagonal(solve_b_a, dim1=-2, dim2=-1).sum(dim=-1)
                  + torch.diagonal(solve_a_b, dim1=-2, dim2=-1).sum(dim=-1))
    logdet_a = torch.linalg.slogdet(cov_a)[1]
    logdet_b = torch.linalg.slogdet(cov_b)[1]
    # The log determinants cancel after symmetrisation, but retaining the
    # explicit form documents the Gaussian KL construction and protects the
    # implementation if the dimension is changed later.
    logdet_term = (logdet_b - logdet_a) + (logdet_a - logdet_b)
    value = 0.25 * (trace_term + mean_term_a + mean_term_b - 4.0
                    + logdet_term)
    return torch.nan_to_num(value, nan=float('inf'), posinf=float('inf'),
                            neginf=0.0).clamp_min(0.0)


def gaussian_wasserstein_distance(
        candidates: torch.Tensor, targets: torch.Tensor,
        eps: float = 1e-6) -> torch.Tensor:
    """Return the Gaussian 2-Wasserstein distance for candidate/target pairs."""
    mean_a, mean_b, cov_a, cov_b = _gaussian_pair(candidates, targets, eps)
    if mean_a.shape[0] == 0:
        return mean_a.new_zeros((0,))
    eye = torch.eye(2, dtype=cov_a.dtype, device=cov_a.device)
    cov_a = cov_a + eps * eye
    cov_b = cov_b + eps * eye
    det_a = torch.linalg.det(cov_a).clamp_min(eps)
    det_b = torch.linalg.det(cov_b).clamp_min(eps)
    trace_ab = torch.diagonal(
        torch.matmul(cov_a, cov_b), dim1=-2, dim2=-1).sum(dim=-1)
    inner = (trace_ab + 2.0 * torch.sqrt((det_a * det_b).clamp_min(eps)))
    cross_trace = torch.sqrt(inner.clamp_min(eps))
    mean_sq = (mean_a - mean_b).square().sum(dim=-1)
    squared = (mean_sq
               + torch.diagonal(cov_a, dim1=-2, dim2=-1).sum(dim=-1)
               + torch.diagonal(cov_b, dim1=-2, dim2=-1).sum(dim=-1)
               - 2.0 * cross_trace)
    return torch.sqrt(torch.nan_to_num(
        squared, nan=float('inf'), posinf=float('inf'), neginf=0.0
    ).clamp_min(0.0))


def normalized_gaussian_wasserstein_distance(
        candidates: torch.Tensor, targets: torch.Tensor,
        eps: float = 1e-6) -> torch.Tensor:
    """Return GWD normalized by the target OBB diagonal length.

    This is an explicitly defined scale-normalized GWD surrogate.  It is
    named ``normalized_gaussian_wasserstein_distance`` rather than ``NWD`` so
    it is not confused with any paper-specific NWD definition.
    """
    candidates, targets = _broadcast_box_pair(candidates, targets)
    distance = gaussian_wasserstein_distance(
        candidates, targets, eps=eps)
    diagonal = torch.sqrt(
        targets[:, 2].abs().square() + targets[:, 3].abs().square()
    ).clamp_min(eps)
    return distance / diagonal

