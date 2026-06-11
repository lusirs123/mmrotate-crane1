# crane_symeood_cm_daga_w075_lowthr_diag.py
# Diagnostic-only config:
#   Use the trained CM-DAGA-W075 checkpoint with a lower inference score_thr
#   to check whether long real-domain empty segments are caused by confidence
#   filtering. Do not use this config for official fair comparison.

_base_ = ['crane_symeood_cm_daga_w075.py']

model = dict(
    bbox_head=dict(
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.001,
            nms=dict(iou_thr=0.1),
            max_per_img=1)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(iou_thr=0.1),
        max_per_img=1),
)

work_dir = 'work_dirs/crane_symeood_cm_daga_w075_lowthr_diag'
