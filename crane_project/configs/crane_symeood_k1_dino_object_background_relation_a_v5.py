"""A: 24-epoch K1-initialized detection-only control."""

_base_ = ['./crane_symeood_k1_dino_object_background_relation_common_v5.py']

model = dict(object_background_relation=None)
work_dir = 'work_dirs/crane_symeood_k1_dino_object_background_relation_a_v5'
