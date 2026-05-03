# =========================================================
# 对比实验 A：经典旋转检测基线 (Rotated RetinaNet)
# 学术定位：展示 NMS 机制与离散 IoU 分配引发的时序抖动缺陷
# =========================================================


_base_ = [
    './_base_/datasets/crane_dota.py',   # 挂载你自定义的单类抓斗数据流
    'mmrotate::_base_/models/retinanet_obb_r50_fpn.py',
    'mmrotate::_base_/schedules/schedule_1x.py',# 继承官方标准的训练基座，随后下方覆盖为 24 Epoch
    'mmrotate::_base_/default_runtime.py',# 继承官方标准的可视化与日志运行环境
]

# _base_ = [
#     '/root/mmrotate/crane_project/configs/_base_/datasets/crane_dota.py',
# # 以下两个引用官方路径，不要用相对路径，直接用绝对路径对齐
#     '/root/mmrotate/configs/_base_/schedules/schedule_1x.py',
#     '/root/mmrotate/configs/_base_/default_runtime.py'
# ]

angle_version = 'le90'

model = dict(
    type='RotatedRetinaNet',  # 不支持.引用
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    
    # --- 躯干：与 SymEOOD 保持绝对一致 ---
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
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

    # =========================================================
    # [头部设定]：传统独立双分支架构
    # =========================================================
    bbox_head=dict(
        type='RotatedRetinaHead',  # MMRotate 注册名是 RotatedRetinaHead，不带 mmrotate. 前缀
        num_classes=1,                      # 抓斗单目标
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            # 必须与你的 SymEOOD 保持一致的极端长宽比先验
            ratios=[0.2, 0.5, 2.0, 5.0], 
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        
        # --- 传统各向同性惩罚（缺乏几何嗅探） ---
        loss_cls=dict(
            type='FocalLoss',  # 原生 Focal Loss，无拓扑排他性压制
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='RotatedIoULoss',   # 传统旋转 IoU 损失，易受极值截断影响
            mode='linear',
            loss_weight=1.0)
    ),

    # =========================================================
    # [训练与推理拓扑]：传统一对多分配与启发式 NMS
    # =========================================================
    
    train_cfg=dict(
        # 传统的离散阈值匹配
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
        # [核心反面教材] NMS 强制开启：离散排序逻辑是引发角度抖动的元凶
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=1)
)

# ─────────────────────────────────────────────────────────
# 运行时 / 训练调度（与 SymEOOD 保持一致）
# ─────────────────────────────────────────────────────────
max_epochs = 24

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# ─────────────────────────────────────────────────────────
# 运行时配置（与 SymEOOD 的 checkpoint 选择对齐）
# ─────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=5,
        save_best='crane/sim/ACI',
        rule='greater',
    ),
)

work_dir = 'work_dirs/crane_baseline'