# =========================================================
# 对比实验 A：经典旋转检测基线 (Rotated RetinaNet)
# 学术定位：展示 NMS 机制与离散 IoU 分配引发的时序抖动缺陷
# =========================================================
# 同时注意 .DS_Store 文件（macOS 系统文件）混入了 annfiles/ 目录，需要清理，否则解析时可能触发异常：



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
    )
)

# =========================================================
# Pipeline 硬编码（从基类提取，彻底消除继承歧义）
# =========================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version='le90'),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad', size_divisor=32),
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
             dict(type='Pad', size_divisor=32),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img']),
         ]),
]

# =========================================================
# 数据流形
# =========================================================
dataset_type = 'CraneDataset'
data_root = '/root/EOOD/crane_project/data/crane_grab/'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
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
# 优化器与评估
# =========================================================
runner = dict(type='EpochBasedRunner', max_epochs=24)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# evaluation = dict(
#     interval=2,
#     metric='mAP',
#     save_best='R_center',
#     rule='greater'
# )
# =========================================================
# 评估器劫持与加权决策下发
# =========================================================
evaluation = dict(
    interval=2,
    metric='mAP',
    save_best='Weighted_R_center',  # 强制 EvalHook 追踪加权分
    rule='greater',
    
    # --- 穿透式物理补偿参数 ---
    thresh_sim=10.0,    # 压榨仿真数据 10 像素级拟合极限
    thresh_real=25.0,   # 兼容真实数据 25 像素人眼抖动方差
    weight_sim=0.7,     # 赋予纯净数据高梯度权重，确保拓扑不崩
    weight_real=0.3     # 赋予真实数据低权重，仅作为防止致盲的惩罚项
)
