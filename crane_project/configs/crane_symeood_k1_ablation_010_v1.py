"""S4x ablation group 010: A=0 B=1 C=0.
Engineering definition only; no training launched by this file.
Inherits the historical SymEOOD+scale-B config. Overrides are limited
to the three ablation switches and work_dir.
"""
_base_ = ['./crane_symeood_k1_ablation_v1_base.py']

work_dir = 'work_dirs/symeood_ablation_v1/010'

model = dict(
    bbox_head=dict(
        loss_cls=dict(use_geometry_weight=False),
    ),
    train_cfg=dict(assigner=dict(cost_reg_mode='iou')))
