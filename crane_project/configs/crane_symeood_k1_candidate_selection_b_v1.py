"""K1 warmstart with one direct top-1 classification selection loss."""

_base_ = ['./crane_symeood_k1.py']

model = dict(bbox_head=dict(candidate_selection=dict(
    score_threshold=0.05,
    iou_threshold=0.5,
    margin=0.1,
    loss_weight=0.05)))
load_from = 'work_dirs/crane_symeood_k1/epoch_20.pth'
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_candidate_selection_b_v1'
