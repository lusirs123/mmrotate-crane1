# =========================================================
# 对比实验 C：原版 EOOD（论文主架构）
# 学术定位：作为 SymEOOD 的直接前身，用于消融"高斯流形统一"的有效性。
# 关键差异（vs SymEOOD）：
#   1. 损失：FocalLoss + L1Loss + RotatedIoULoss（vs SymNFL + SymKLD）
#   2. 分配：MaxIoU(冷启动) + PolaAssigner(精修)（vs SymPOLA + 高斯代价）
#   3. 多头融合：3 个并行 predictor（Eood/ATSS/FCOS）
#   4. 推理：仅 predictors[0]（即 RotatedEoodHead）参与
# =========================================================
# 关键工程决策：手搓 SetEpochInfoHook 复刻 MMRotate 1.x 的 epoch 感知机制，
# 避免 init_epoch=0 的"用噪声指导噪声"灾难。前 init_epoch 个 epoch 用 MaxIoU
# 做空间拓扑预热，之后切换为 Pola 做精细化分配。

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.core.hooks.set_epoch_info_hook',
    ],
    allow_failed_imports=False)

_base_ = [
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py',
]

angle_version = 'le90'
max_epochs = 24

# =========================================================
# 模型：原版 EOOD 主结构（4 predictors 并行）
# 适配单类抓斗：num_classes=1, anchor_generator 与 SymEOOD 对齐
# =========================================================
model = dict(
    type='Eood',
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
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='EoodHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        parallel=True,
        predictors=[
            # ─── Predictor 0: RotatedEoodHead（推理唯一参与者）──────────
            dict(
                type='RotatedEoodHead',
                num_classes=1,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                assign_by_circumhbbox=None,
                anchor_generator=dict(
                    type='RotatedAnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=1,
                    ratios=[0.5, 1.0, 2.0],
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
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                loss_iou=dict(
                    type='RotatedIoULoss',
                    loss_weight=2.0,
                    mode='linear',
                    diou=True),
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.4,
                        min_pos_iou=0,
                        ignore_iof_thr=-1,
                        iou_calculator=dict(type='RBboxOverlaps2D')),
                    pola=dict(
                        type='PolaAssigner',
                        cost_class=1.0,
                        cost_bbox=1.0,
                        cost_riou=1.0),
                    init_epoch=4,
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False),
                test_cfg=dict(
                    nms_pre=2000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(iou_thr=0.1),
                    max_per_img=2000)),
            # ─── Predictor 1: RotatedATSSHead（辅助监督）────────────────
            dict(
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
                train_cfg=dict(
                    assigner=dict(
                        type='ATSSObbAssigner',
                        topk=9,
                        angle_version=angle_version,
                        iou_calculator=dict(type='RBboxOverlaps2D')),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False)),
            # ─── Predictor 2: RotatedFCOSHead（Anchor-Free 辅助监督）────
            dict(
                type='RotatedFCOSHead',
                num_classes=1,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                strides=[8, 16, 32, 64, 128],
                center_sampling=True,
                center_sample_radius=1.5,
                norm_on_bbox=True,
                centerness_on_reg=True,
                separate_angle=False,
                scale_angle=True,
                bbox_coder=dict(
                    type='DistanceAnglePointCoder',
                    angle_version=angle_version),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='RotatedIoULoss',
                    loss_weight=1.0,
                    mode='linear',
                    diou=False),
                loss_centerness=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                train_cfg=None),
        ]),
)

# =========================================================
# 数据流形（与 baseline / symeood 严格对齐）
# =========================================================
dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version=angle_version),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad',
         size=(1024, 1024),
         pad_val=dict(img=(114.0, 114.0, 114.0))),
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
                  size=(1024, 1024),
                  pad_val=dict(img=(114.0, 114.0, 114.0))),
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

# =========================================================
# 训练调度（与 baseline / symeood 对齐，公平对比）
# =========================================================
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
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

log_config = dict(interval=100)

# =========================================================
# Hook 注入：epoch 信息感知（驱动 EOOD 内部分配器切换）
# =========================================================
custom_hooks = [
    dict(type='SetEpochInfoHook')
]

log_level = 'INFO'
load_from = None
resume_from = None
work_dir = 'work_dirs/crane_eood'
