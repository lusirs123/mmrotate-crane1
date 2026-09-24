"""Arm C: original DINO feature loss with gradient into the FPN."""

_base_ = ['./crane_symeood_k1_dino_fpn_gradient_common_v3.py']

model = dict(semantic_distillation=dict(protect_geometry=False))

work_dir = 'work_dirs/crane_symeood_k1_dino_fpn_gradient_c_v3'

