"""Corrected-mask comparison A: matched K1 training without DINO loss."""

_base_ = ['./crane_symeood_k1_dino_fpn_gradient_common_v3.py']

model = dict(semantic_distillation=None)

work_dir = 'work_dirs/crane_symeood_k1_dino_corrected_mask_a_v4'
