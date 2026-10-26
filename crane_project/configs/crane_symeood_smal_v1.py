_base_ = ['./crane_symeood_mal_v1_base.py']
model = dict(bbox_head=dict(loss_cls=dict(objective='S_MAL')))
work_dir = 'work_dirs/symeood_mal_strength_seed0_20261026_v1/S_MAL'
