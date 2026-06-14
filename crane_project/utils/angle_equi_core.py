"""Pure utilities for flip angle-equivariance tests and loss wiring.

This module intentionally depends only on Python stdlib and torch. It must be
importable on machines without mmcv, mmdet, or mmrotate installed.
"""

import math

import torch

__all__ = [
    'angle_to_emb', 'flip_emb', 'equi_flip_loss', 'invar_photo_loss',
    'mirror_grid_indices', 'valid_feat_mask', 'extract_emb_at_positions',
    'build_photo_params', 'apply_t_photo',
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


# ==============================================================================
# T_photo: photometric augmentation on GPU (torch tensor ops)
# ==============================================================================

def build_photo_params(B: int, device: torch.device,
                       gamma_range=(0.7, 1.5),
                       rg_range=(0.95, 1.40),
                       bg_range=(0.75, 1.05),
                       grad_lr_range=(0.5, 2.0),
                       grad_ud_range=(0.7, 1.5),
                       contrast_range=(0.5, 2.0),
                       on_prob: float = 0.5,
                       ) -> dict:
    """Sample per-image T_photo parameters (Route B: in-distribution).

    Each component is independently turned on with probability ``on_prob``.
    When off, the sampled value is replaced by the identity (=1.0), so the
    component contributes no perturbation for that image.
    """
    def _sample(rng):
        v = torch.empty(B, device=device).uniform_(*rng)
        m = torch.bernoulli(torch.full((B,), on_prob, device=device))
        return torch.where(m > 0.5, v, torch.ones_like(v))

    return dict(
        gamma=_sample(gamma_range),
        rg_gain=_sample(rg_range),
        bg_gain=_sample(bg_range),
        contrast=_sample(contrast_range),
        grad_lr=_sample(grad_lr_range),
        grad_ud=_sample(grad_ud_range),
    )


# ImageNet RGB Normalize constants used by the training pipeline.
# Pipeline order: to_rgb=True -> Normalize(mean, std) so the tensor seen here
# is in RGB order with values approximately in [-2.12, +2.64].
_PHOTO_MEAN_RGB = (123.675, 116.28, 103.53)
_PHOTO_STD_RGB = (58.395, 57.12, 57.375)


def _denormalize_rgb(img: torch.Tensor) -> torch.Tensor:
    """Map a Normalize'd RGB tensor back to [0, 1] linear-light space."""
    mean = img.new_tensor(_PHOTO_MEAN_RGB).view(1, 3, 1, 1) / 255.0
    std = img.new_tensor(_PHOTO_STD_RGB).view(1, 3, 1, 1) / 255.0
    return img * std + mean


def _renormalize_rgb(img: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_denormalize_rgb`."""
    mean = img.new_tensor(_PHOTO_MEAN_RGB).view(1, 3, 1, 1) / 255.0
    std = img.new_tensor(_PHOTO_STD_RGB).view(1, 3, 1, 1) / 255.0
    return (img - mean) / std


def apply_t_photo(img: torch.Tensor, params: dict) -> torch.Tensor:
    """Apply T_photo augmentation to a batch of images (GPU torch op).

    Operates in linear [0, 1] RGB space:
      Normalize'd input -> denormalize -> gamma -> ch_gain (RGB) ->
      spatial gradient -> contrast -> clip -> renormalize.

    Args:
        img: (B, 3, H, W) Normalize'd tensor (RGB order).
        params: dict from :func:`build_photo_params`.

    Returns:
        (B, 3, H, W) augmented tensor in the same Normalize'd space.
    """
    B, C, H, W = img.shape

    x = _denormalize_rgb(img).clamp(0.0, 1.0)

    # 1. Gamma in linear [0, 1] space.
    gamma = params['gamma'].view(B, 1, 1, 1)
    x = x.clamp(min=1e-6) ** gamma

    # 2. Per-channel gain in RGB order (channel 0 = R, channel 2 = B).
    ch_gain = torch.stack([
        params['rg_gain'],                      # R / G ratio
        torch.ones(B, device=img.device),       # G = reference
        params['bg_gain'],                      # B / G ratio
    ], dim=1).view(B, 3, 1, 1)
    x = x * ch_gain

    # 3. Spatial gradient (left-right and up-down brightness variation).
    lr = params['grad_lr'].view(B, 1, 1, 1)
    ud = params['grad_ud'].view(B, 1, 1, 1)
    t_lr = torch.linspace(0, 1, W, device=img.device).view(1, 1, 1, W)
    t_ud = torch.linspace(0, 1, H, device=img.device).view(1, 1, H, 1)
    grad = (1.0 + (lr - 1.0) * t_lr) * (1.0 + (ud - 1.0) * t_ud)
    x = x * grad

    # 4. Contrast adjustment around the per-image mean.
    mean_val = x.mean(dim=(2, 3), keepdim=True)
    contrast = params['contrast'].view(B, 1, 1, 1)
    x = mean_val + (x - mean_val) * contrast

    # 5. Clip to valid linear range, re-encode to Normalize'd space.
    x = x.clamp(0.0, 1.0)
    return _renormalize_rgb(x)


# ==============================================================================
# L_invar: photometric invariance loss (no mirror needed)
# ==============================================================================

def invar_photo_loss(pred_emb_orig: torch.Tensor,
                     pred_emb_photo: torch.Tensor) -> torch.Tensor:
    """Photometric invariance loss in embedding space (one-directional).

    Constrains emb(f(T_photo(x))) -> sg(emb(f(x))) so the original image
    branch keeps its detection-driven gradients (protecting A-RMSE) while
    the perturbed branch is pulled toward the original.
    """
    if pred_emb_orig.numel() == 0:
        return pred_emb_orig.new_zeros(())

    diff = pred_emb_photo - pred_emb_orig.detach()
    loss_per_pair = (diff * diff).sum(dim=-1)

    if loss_per_pair.numel() == 0:
        return pred_emb_orig.new_zeros(())

    return loss_per_pair.mean()
