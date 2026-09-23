"""Matched K1-initialized control with no DINO loss."""

_base_ = ['./crane_symeood_k1_dino_warmstart_common_v2.py']

# Keep the same classification adapter and cached-feature data pipeline.
# The detector accepts teacher_features but does not use them in this arm.
model = dict(semantic_distillation=None)

work_dir = 'work_dirs/crane_symeood_k1_dino_warmstart_control_v2'

