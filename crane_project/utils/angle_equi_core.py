"""Pure utilities for flip angle-equivariance tests and loss wiring.

This module intentionally depends only on Python stdlib and torch. It must be
importable on machines without mmcv, mmdet, or mmrotate installed.
"""

import math

import torch

__all__ = [
    'angle_to_emb', 'flip_emb', 'equi_flip_loss',
    'mirror_grid_indices', 'valid_feat_mask', 'extract_emb_at_positions'
]


def angle_to_emb(theta: torch.Tensor) -> torch.Tensor:
    """Map le90 angle theta to the 180-degree periodic embedding."""
    two_theta = 2.0 * theta
    return torch.stack([torch.cos(two_theta), torch.sin(two_theta)], dim=-1)


def flip_emb(emb: torch.Tensor, direction: str) -> torch.Tensor:
    """Apply the le90 flip transform in embedding space."""
    if direction in ('horizontal', 'vertical'):
        return emb * emb.new_tensor([1.0, -1.0])
    if direction == 'diagonal':
        return emb
    raise ValueError(
        f"Unknown direction: {direction}. Expected 'horizontal', "
        "'vertical', or 'diagonal'.")


def mirror_grid_indices(row: torch.Tensor, col: torch.Tensor,
                        H_feat: int, W_feat: int,
                        direction: str) -> tuple:
    """Integer grid mirror mapping, with no interpolation."""
    if direction == 'horizontal':
        return row, W_feat - 1 - col
    if direction == 'vertical':
        return H_feat - 1 - row, col
    if direction == 'diagonal':
        return H_feat - 1 - row, W_feat - 1 - col
    raise ValueError(f"Unknown direction: {direction}.")


def valid_feat_mask(row: torch.Tensor,
                    col: torch.Tensor,
                    H_feat: int,
                    W_feat: int,
                    img_H: int,
                    img_W: int,
                    pad_H: int,
                    pad_W: int) -> torch.Tensor:
    """Mask source feature cells inside the left/top unpadded image region."""
    if row.numel() == 0:
        return row.new_zeros(row.shape, dtype=torch.bool)

    valid_h = min(H_feat, int(math.ceil(float(img_H) * H_feat / float(pad_H))))
    valid_w = min(W_feat, int(math.ceil(float(img_W) * W_feat / float(pad_W))))
    return ((row >= 0) & (row < valid_h) &
            (col >= 0) & (col < valid_w))


def extract_emb_at_positions(emb_map: torch.Tensor,
                             row: torch.Tensor,
                             col: torch.Tensor) -> torch.Tensor:
    """Gather embedding values from a (2, H, W) map at integer positions."""
    return emb_map[:, row, col].t()


def equi_flip_loss(pred_emb_orig: torch.Tensor,
                   pred_emb_flip: torch.Tensor,
                   direction: str,
                   valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Strong flip-equivariance loss in embedding space."""
    if pred_emb_orig.numel() == 0:
        return pred_emb_orig.new_zeros(())

    target_emb = flip_emb(pred_emb_orig, direction)
    loss_per_pair = ((pred_emb_flip - target_emb) ** 2).sum(dim=-1)

    if valid_mask is not None and valid_mask.numel() > 0:
        loss_per_pair = loss_per_pair[valid_mask]

    if loss_per_pair.numel() == 0:
        return pred_emb_orig.new_zeros(())

    return loss_per_pair.mean()
