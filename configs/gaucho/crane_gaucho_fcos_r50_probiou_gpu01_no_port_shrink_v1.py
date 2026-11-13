"""Paired GauCho control without the project's training-scale shrink."""
_base_ = ['./crane_gaucho_fcos_r50_probiou_gpu01_observed_v1.py']

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(1024, 1024),
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# MMCV replaces the train dataset list, so both domains need the new pipeline.
data = dict(train=[
    dict(type='CraneDataset', classes=('grab',),
         data_root='crane_project/data/crane_grab_port_day2night_v1/',
         ann_file='train/annfiles/', img_prefix='train/images/',
         pipeline=train_pipeline, version='le90'),
    dict(type='CraneDataset', classes=('grab',),
         data_root='crane_project/data/crane_grab_port_day2night_v1/',
         ann_file='train_sim/annfiles/', img_prefix='train_sim/images/',
         pipeline=train_pipeline, version='le90'),
])
work_dir = 'work_dirs/gaucho_no_port_shrink_v1/attempt_001/gaucho/train'
