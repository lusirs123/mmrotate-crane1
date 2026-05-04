# crane_project/configs/_base_/datasets/crane_dota.py
# 纯净 0.x 语法版本

dataset_type = 'CraneDataset'  # <-- 核心：调用你刚刚在底层注册的自定义类
data_root = '/root/mmrotate/crane_project/data/crane_grab/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 0.x 时代的纯粹数据管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(704, 576)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(704, 576),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# 0.x 时代的核心数据分发字典
data = dict(
    samples_per_gpu=4,  # batch_size
    workers_per_gpu=4,  # num_workers
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline)
)