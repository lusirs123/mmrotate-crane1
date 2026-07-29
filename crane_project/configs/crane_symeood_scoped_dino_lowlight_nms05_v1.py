"""Scoped frozen-DINO rescue with source-selected ROI NMS=0.5.

The BrightAug SymEOOD branch is unchanged and still uses its custom no-NMS
top-1 inference.  Only the independent DINO rescue ROI head uses this more
permissive rotated-NMS threshold, which retains overlapping small-object
rescue candidates for downstream ordering/post-processing.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_v1.py']

# Keep the standalone provenance dictionary and the deployable model in sync.
dino_rescue = dict(
    head=dict(roi_nms_iou_thr=0.5))

model = dict(
    dino_rescue=dict(
        head=dict(roi_nms_iou_thr=0.5)))
