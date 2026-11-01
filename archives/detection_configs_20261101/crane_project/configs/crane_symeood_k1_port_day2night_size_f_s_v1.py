"""F-S preflight candidate: B + independent log-size SmoothL1, beta/lambda=.1.

Only bounded checks are authorized. Not approved for formal training yet.
Same original ImageNet initialization/budget; no B resume, D or H stacking.
"""
_base_ = ['./crane_symeood_k1_port_day2night_aug_b_v1.py']

model = dict(bbox_head=dict(shape_compensation=dict(
    type='LogSizeLoss', beta=0.1, loss_weight=0.1, eps=1e-6, reduction='mean')))

work_dir = 'work_dirs/crane_symeood_k1_port_day2night_size_f_s_v1'
