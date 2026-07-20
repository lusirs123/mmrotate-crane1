"""Coordinate-safe padding-candidate filter for checkpoint-only diagnosis.

This config changes neither image preprocessing nor model parameters.  It
only excludes candidates whose anchor center lies in the padded area outside
``img_shape``.  Existing BrightAug checkpoints can therefore be evaluated
directly without retraining.
"""

_base_ = ['./crane_symeood_k1_brightaug.py']

model = dict(
    bbox_head=dict(filter_padding_anchors=True))

load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_brightaug_valid_content'
