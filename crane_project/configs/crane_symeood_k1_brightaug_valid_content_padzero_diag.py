"""Inference-only zero-padding diagnosis with coordinate-safe filtering.

The resized image content and all geometry metadata stay unchanged.  Only the
post-normalization padding value changes from the erroneous 114 to normalized
zero.  Do not use this diagnostic config for training because its inherited
training pipeline intentionally remains untouched.
"""

_base_ = ['./crane_symeood_k1_brightaug_valid_content.py']

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='Pad',
                size=(1024, 1024),
                pad_val=dict(img=(0.0, 0.0, 0.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_brightaug_valid_content_padzero_diag'
