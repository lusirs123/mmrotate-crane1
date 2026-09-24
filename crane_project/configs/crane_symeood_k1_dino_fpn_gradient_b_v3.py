"""Arm B: original DINO feature loss, detached from the FPN."""

_base_ = ['./crane_symeood_k1_dino_fpn_gradient_common_v3.py']

model = dict(semantic_distillation=dict(protect_geometry=True))

work_dir = 'work_dirs/crane_symeood_k1_dino_fpn_gradient_b_v3'

