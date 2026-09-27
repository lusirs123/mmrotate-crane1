"""K1 warmstart control for source candidate selection supervision."""

_base_ = ['./crane_symeood_k1.py']

model = dict(bbox_head=dict(candidate_selection=None))
load_from = 'work_dirs/crane_symeood_k1/epoch_20.pth'
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_candidate_selection_a_v1'
