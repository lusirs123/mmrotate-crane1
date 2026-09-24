"""Arm A: frozen backbone, trainable FPN/head/adapter, no DINO loss."""

_base_ = ['./crane_symeood_k1_dino_fpn_gradient_common_v3.py']

model = dict(semantic_distillation=None)

work_dir = 'work_dirs/crane_symeood_k1_dino_fpn_gradient_a_v3'

