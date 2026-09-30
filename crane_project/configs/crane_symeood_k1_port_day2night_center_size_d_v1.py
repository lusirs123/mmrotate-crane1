"""D = SymEOOD scale B + one positive-only centre/size regression objective.

Inherit data, scale, ImageNet initialization, optimizer and 24-epoch schedule.
The completed B experiment is the control. No DINO or inference changes.
"""
_base_ = ['./crane_symeood_k1_port_day2night_aug_b_v1.py']

model = dict(bbox_head=dict(center_size_compensation=dict(
    type='CenterSizeCompensationLoss', beta=0.1, loss_weight=0.25,
    eps=1e-6, reduction='mean')))

work_dir = 'work_dirs/crane_symeood_k1_port_day2night_center_size_d_v1'
