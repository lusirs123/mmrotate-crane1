# mmrotate/models/losses/angle_equi.py
"""
Flip equivariance + photometric invariance angle losses.

Re-exports pure-torch utilities from crane_project.utils.angle_equi_core
so that mmrotate modules can import them via the standard losses path.
"""

from crane_project.utils.angle_equi_core import (  # noqa: F401
    angle_to_emb, equi_flip_loss, extract_emb_at_positions, flip_emb,
    invar_photo_loss, mirror_grid_indices, valid_feat_mask,
    build_photo_params, apply_t_photo)

__all__ = [
    'angle_to_emb', 'flip_emb', 'equi_flip_loss', 'invar_photo_loss',
    'mirror_grid_indices', 'valid_feat_mask', 'extract_emb_at_positions',
    'build_photo_params', 'apply_t_photo',
]