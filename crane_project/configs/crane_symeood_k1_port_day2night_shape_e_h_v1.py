"""E-H v1 = scale B + fixed 0.05 covariance-only smoothed Hellinger loss.

Same ImageNet initialization, seed0, TRAIN/VAL, optimizer and 24 epochs as B.
Do not resume from B or add D centre compensation. Formula/eps/precision are
frozen before training; this is a project adaptation, not full ProbIoU.
"""
_base_ = ['./crane_symeood_k1_port_day2night_aug_b_v1.py']

model = dict(bbox_head=dict(shape_compensation=dict(
    type='CovarianceShapeLoss', mode='hellinger', loss_weight=0.05,
    eps=1e-6, reduction='mean')))

work_dir = 'work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1'
