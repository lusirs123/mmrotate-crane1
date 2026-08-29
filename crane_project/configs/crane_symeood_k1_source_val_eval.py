"""Evaluation-only source-val view of the formally selected ordinary K1.

This config exists only to produce the 738-frame source-val reference PKL for
the causal-history gate.  It does not train, read fixed TEST, or select a new
K1 checkpoint.  The checkpoint remains the existing source-selected epoch 24.
"""

_base_ = ['./crane_symeood_k1.py']

source_val_pipeline = [
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
                type='Pad', size=(1024, 1024),
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

source_val_dataset = dict(
    type='CraneDataset',
    data_root='crane_project/data/crane_grab/',
    ann_file='val/annfiles/',
    img_prefix='val/images/',
    pipeline=source_val_pipeline,
    version='le90')

data = dict(
    _delete_=True,
    val=source_val_dataset,
    test=source_val_dataset,
    train_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

load_from = None
resume_from = None
