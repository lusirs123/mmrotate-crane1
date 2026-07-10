# =========================================================
# SymEOOD(K=1) + AuxDetachClsHead — 梯度隔离辅助分类头
#
# 设计原理：
#   主头在 dead segments 上几何已完美（RIoU 0.620, 33/33 usable），
#   唯一问题是 cls confidence 太低（global_max ≈ 0.006-0.01）。
#
#   aux_detach_cls_head 训练在 strongly augmented + DETACHED FPN features
#   上，学习"困难帧也有高置信度"。detach() 确保梯度从不回传到
#   backbone/FPN，主头训练完全不受影响。
#
#   推理时：同一张干净图 → main head (reg+angle) + aux_cls_head (cls)
#           cls_fused = max(main_cls_logits, aux_cls_logits)
#           bbox始终来自主头 → 几何零退化
#
# 与历史方案的对比：
#   - aux2_sgldet: 衰减（低 loss_weight），梯度仍回传 → 1.695 退化
#   - 外扩头 injection: 无隔离 → 0.620→0.143 几何崩
#   - scorectx: 加性偏置 → gate 膨胀/boost 太小
#   - QFL: 改 loss target → 训练域退化
#   - 本方案: detach FPN → 硬隔离，主头正常训练，几何零退化
# =========================================================

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.aux_detach_cls_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

_base_ = [
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py',
]

angle_version = 'le90'
max_epochs = 24

# =========================================================
# 模型：SymEOOD(K=1) + AuxDetachClsHead
# =========================================================
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
        type='SymEOODHead',
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
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='retina_cls',
                std=0.01,
                bias_prob=0.01)),

        # 创新点 1+2：SymNFLLoss
        loss_cls=dict(
            type='SymNFLLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            tau_init=10.0,
            tau_min=1.0,
            warmup_iters=1000,
            loss_call_factor=5,
            eps=1e-6,
            reduction='mean',
            loss_weight=0.25,
            kld_chunk_size=256),

        # 创新点 1：SymKLDLoss
        loss_bbox=dict(
            type='SymKLDLoss',
            eps=1e-6,
            reduction='mean',
            loss_weight=2.0)),

    train_cfg=dict(
        assigner=dict(
            type='SymPOLAAssigner',
            cost_class=1.0,
            cost_reg=2.0,
            o2m=False,
            topk=1,
            o2m_warmup_iters=2000,
            o2m_topk=9,
            tau_init=10.0,
            tau_min=1.0,
            warmup_iters=1000,
            eps=1e-6),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=1),

    # K=1 辅助监督：与 k1 baseline 保持等价
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
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='retina_cls',
                std=0.01,
                bias_prob=0.01)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.01),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.01),
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
            max_per_img=1),
    )],

    # ============================================================
    # AuxDetachClsHead: 梯度隔离辅助分类头
    #
    # 训练期：strong aug + detach FPN features → 只训练 aux_cls_head
    #   梯度从不回传到 backbone/FPN → 主头训练零干扰
    #
    # 推理期：同一张干净图 → max(main_cls, aux_cls) fusion
    #   bbox 始终来自主头 → 几何零退化
    # ============================================================
    aux_detach_cls_head=dict(
        type='AuxDetachClsHead',
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        num_classes=1,
        num_anchors=3,  # ratios=[0.5, 1.0, 2.0], 1 scale

        # FocalLoss for binary FG/BG classification
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        # IoU-based anchor assignment (independent of main head's SymPOLA)
        # Simple threshold assignment is sufficient for cls-only training
        aux_detach_pos_iou_thr=0.5,
        aux_detach_neg_iou_thr=0.4,

        # Strong augmentation params
        aux_detach_gamma_range=(0.1, 0.8),       # aggressive dimming
        aux_detach_blur_sigma_range=(0.5, 3.0),   # gaussian blur
        aux_detach_blur_kernel=7,
        aux_detach_noise_std_range=(0.0, 30.0),   # sensor noise /255
        aux_detach_downscale_range=(0.5, 0.8),    # down+upsample
        aux_detach_contrast_range=(0.5, 1.5),
        aux_detach_rg_range=(0.7, 1.3),
        aux_detach_bg_range=(0.7, 1.3),
    ),
)

# =========================================================
# 数据流形（与 k1 严格对齐）
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
# 训练调度（与 k1 对齐）
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

# Warm start from k1 epoch_24 (main head geometry is already optimal)
load_from = 'work_dirs/crane_symeood_k1/epoch_24.pth'

checkpoint_config = dict(interval=2, max_keep_ckpts=24)
evaluation = dict(
    interval=2,
    metric='mAP',
    save_best='Weighted_R_center',
    rule='greater',
    thresh_sim=10.0,
    thresh_real=25.0,
    weight_sim=0.7,
    weight_real=0.3)

log_config = dict(interval=50)

log_level = 'INFO'
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_auxdetach'
