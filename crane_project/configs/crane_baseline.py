# =========================================================
# 对比实验 A：经典旋转检测基线 (Rotated RetinaNet)
# 学术定位：展示 NMS 机制与离散 IoU 分配引发的时序抖动缺陷
# =========================================================
# 同时注意 .DS_Store 文件（macOS 系统文件）混入了 annfiles/ 目录，需要清理，否则解析时可能触发异常：

#注意防止网络中断导致重启，使用会话
# tmux new -s crane_train 
#tmux attach -t crane_train 重新使用原本的

# 启动训练
# 

_base_ = [
    '../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'
]

# =========================================================
# 算子覆写区
# =========================================================
model = dict(
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            type='RotatedAnchorGenerator', 
            ratios=[0.2, 0.5, 1.0, 2.0, 5.0],
            strides=[8, 16, 32, 64, 128]
        )
    ),
        # ================= 强制覆写基类测试配置 =================
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,  # [核心修改] 将默认 0.05 降至极限底噪
        nms=dict(iou_thr=0.1),
        max_per_img=2000
        ),
)

# =========================================================
# Pipeline 硬编码（从基类提取，彻底消除继承歧义）
# =========================================================
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
         pad_val=dict(img=(114.0, 114.0, 114.0))),     # ← 灰色填充（114 是 ImageNet 均值近似）
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
                  pad_val=dict(img=(114.0, 114.0, 114.0))),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img']),
         ]),
]
# =========================================================
# 数据流形
# =========================================================
dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'


# 使用两种 1080 显卡，

data = dict(
    samples_per_gpu=2, #samples_per_gpu=4 与双卡（2 张 1080）共同决定了整个训练过程的总批次大小，保证总批次大小一致即可
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train/annfiles/',        # ← 目录，不是 .txt
            img_prefix='train/images/',
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train_sim/annfiles/',    # ← 目录
            img_prefix='train/images/',
            pipeline=train_pipeline,
        ),
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',              # ← 目录
        img_prefix='val/images/',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/annfiles/',             # ← 目录
        img_prefix='test/images/',
        pipeline=test_pipeline,
    ),
)
# =========================================================
# 优化器与训练调度（与 crane_symeood 严格对齐，确保对比公平）
# =========================================================
runner = dict(type='EpochBasedRunner', max_epochs=24)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

# 梯度截断：对齐 symeood 的 max_norm=10（基类默认 35，对小数据集过松）
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# 学习率策略：对齐 symeood 的 1k warmup + step=[16, 22]（基类默认 500/[8, 11] 仅适配 12 epochs）
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 22])

# checkpoint：对齐 symeood interval=2、max_keep_ckpts=5
checkpoint_config = dict(interval=2, max_keep_ckpts=5)

# =========================================================
# 评估器劫持与加权决策下发（权重对齐 symeood：sim=0.7 / real=0.3）
# =========================================================
evaluation = dict(
    interval=2,
    metric='mAP',
    save_best='Weighted_R_center',  # 强制 EvalHook 追踪加权分
    rule='greater',

    # --- 穿透式物理补偿参数 ---
    thresh_sim=10.0,    # 压榨仿真数据 10 像素级拟合极限
    thresh_real=25.0,   # 兼容真实数据 25 像素人眼抖动方差
    weight_sim=0.7,     # 与 symeood 对齐
    weight_real=0.3     # 与 symeood 对齐
)

# 日志间隔与 symeood 对齐
log_config = dict(interval=100)

log_level = 'INFO'
load_from = None
resume_from = None
work_dir = 'work_dirs/crane_baseline'
