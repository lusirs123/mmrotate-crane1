"""S4x ablation group 000: A=0 B=0 C=0.
Engineering definition only; no training launched by this file.
Inherits the historical SymEOOD+scale-B config. Overrides are limited
to the three ablation switches and work_dir.
"""
_base_ = ['./crane_symeood_k1_ablation_v1_base.py']

work_dir = 'work_dirs/symeood_ablation_v1/000'

model = dict(
    bbox_head=dict(
        loss_cls=dict(use_geometry_weight=False),
        loss_bbox=dict(_delete_=True, type='RotatedIoULoss',
                       mode='linear', diou=True, loss_weight=2.0,
                       eps=1e-06, reduction='mean'),
    ),
    train_cfg=dict(assigner=dict(cost_reg_mode='iou')))
