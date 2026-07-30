"""Protected S7 proposal supplement for the frozen-DINO detector.

The native formal low-light config remains unchanged.  This config is used
only after the source-only S7 checkpoint has passed its retention gate.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_v1.py']

_s7_head = dict(
    gpu=0,
    rpn_feat_channels=256,
    roi_fc_channels=1024,
    roi_samples=256,
    proposal_count=2000,
    max_detections=2000,
    roi_nms_iou_thr=0.5,
    s7_residual=True,
    s7_channels=128,
    s7_rpn_feat_channels=128,
    s7_proposal_count=500,
    s7_nms_pre=2000,
    s7_anchor_sizes=[16.0, 32.0, 64.0, 128.0, 256.0])

model = dict(
    dino_rescue=dict(head=_s7_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_residual_v1/'
        'labeller_best_source_only.pth'))

dino_rescue = dict(head=_s7_head)
