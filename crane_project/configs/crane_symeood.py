# crane_symeood.py
# MMRotate 0.x 配置：SymEOOD baseline + CraneDataset + symKLD + symNFL
# 使用根目录的 tools/train.py / tools/test.py。

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
    ],
    allow_failed_imports=False)

_base_ = [
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py',
]

angle_version = 'le90'
max_epochs = 24

model = dict(
    type='SymEOOD',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='SymEOODHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.2, 0.5, 2.0, 5.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='SymNFLLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            tau_init=10.0,
            tau_min=1.0,
            warmup_iters=500,
            eps=1e-6,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='SymKLDLoss',
            eps=1e-6,
            reduction='mean',
            loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=1)),
    aux_bbox_head=[dict(
        type='RotatedATSSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='ATSSObbAssigner',
                topk=9,
                angle_version=angle_version,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=1))],
)

# =========================================================
# 数据流形（显式写出，避免依赖外部 base 数据配置）
# =========================================================
dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),  # keep_ratio=True，等比缩放
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version='le90'),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad',
         size=(1024, 1024),          # ← 从 size_divisor=32 改为绝对尺寸
         pad_val=dict(img=114)),     # ← 灰色填充（114 是 ImageNet 均值近似）
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1024, 1024),
         flip=False,
         transforms=[
             dict(type='RResize'),
             dict(type='Normalize',
                  mean=[123.675, 116.28, 103.53],
                  std=[58.395, 57.12, 57.375],
                  to_rgb=True),
             dict(type='Pad',
                  size=(1024, 1024),      # ← 与训练完全一致
                  pad_val=dict(img=114)),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img']),
         ]),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train/annfiles/',
            img_prefix='train/images/',
            pipeline=train_pipeline,
            version=angle_version),
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train_sim/annfiles/',
            img_prefix='train/images/',
            pipeline=train_pipeline,
            version=angle_version),
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        img_prefix='val/images/',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/annfiles/',
        img_prefix='test/images/',
        pipeline=test_pipeline,
        version=angle_version),
)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=2, max_keep_ckpts=5)
evaluation = dict(
    interval=2,
    metric='mAP',
    save_best='Weighted_R_center',
    rule='greater',
    thresh_sim=10.0,
    thresh_real=25.0,
    weight_sim=0.7,
    weight_real=0.3)

log_level = 'INFO'
load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood'
