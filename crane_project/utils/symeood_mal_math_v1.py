"""Frozen scalar arithmetic for MAL / its own-forward strength control."""
import torch
import torch.nn.functional as F


def mal_terms(logits, labels, quality, weight, gamma=1.5):
    """Single-class MAL. q and the negative probability weight are detached."""
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError('Single-class logits required')
    if labels.shape != quality.shape or labels.shape != weight.shape:
        raise ValueError('Unaligned targets')
    if not torch.isfinite(logits).all() or not torch.isfinite(quality).all():
        raise FloatingPointError('Nonfinite logits/quality')
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise FloatingPointError('Invalid weights')
    if (quality < -1e-6).any() or (quality > 1+1e-6).any():
        raise FloatingPointError('Quality outside tolerance')
    positive = labels == 0
    target = torch.where(positive, quality.detach().clamp(0, 1).pow(gamma),
                         torch.zeros_like(quality))[:, None]
    modulation = torch.where(positive[:, None], torch.ones_like(logits),
                             logits.sigmoid().detach().pow(gamma))
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            * modulation)[:, 0] * weight


def match_strength(U, M, floor=1e-12, cf_zero_tolerance=1e-12):
    if U.ndim or M.ndim or not torch.isfinite(U) or not torch.isfinite(M):
        raise FloatingPointError('Nonfinite/non-scalar objectives')
    if U.detach() < 0 or M.detach() < 0:
        raise FloatingPointError('Negative objectives')
    degenerate = bool(U.detach() <= floor)
    if degenerate:
        if M.detach() > cf_zero_tolerance:
            raise FloatingPointError('U near zero but counterfactual MAL nonzero')
        alpha = U.detach().new_tensor(1.)
    else:
        alpha = M.detach() / U.detach()
        if not torch.isfinite(alpha) or alpha < 0:
            raise FloatingPointError('Invalid alpha; no clipping allowed')
    return alpha, alpha * U, degenerate
