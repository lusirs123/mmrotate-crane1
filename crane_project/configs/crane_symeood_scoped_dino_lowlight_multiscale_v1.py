"""Experimental frozen-DINO multi-scale candidate-coverage variant.

This keeps the formal BrightAug and DINO checkpoints unchanged while exposing
three interpolation-only patch-grid levels to the DINO RPN/ROI heads.  The
default formal config remains single-scale; this variant must pass source-only
coverage and retention checks before it is used for any target diagnosis.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_nms05_v1.py']

model = dict(
    dino_rescue=dict(
        head=dict(feature_strides=[7, 14, 28])))

dino_rescue = dict(
    head=dict(feature_strides=[7, 14, 28]))
