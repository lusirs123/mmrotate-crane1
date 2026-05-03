# crane_symeood.py
# MMRotate 完整配置文件：EOOD baseline + CraneOBBMetric + symKLD +symPOLA + symNFl 
# 最后实现一对一匹配无 NMS 和度量同构，改善了原始论文 EOOD 的缺点
# 支持非等间隔抽帧的时序评估
#
# 使用方法：
#   单 GPU 训练：python tools/train.py configs/crane_symeood.py
#   多 GPU 训练：bash tools/dist_train.sh configs/crane_symeood.py 4
#   测试评估：   python tools/test.py configs/crane_symeood.py \
#                   work_dirs/crane_symeood/latest.pth
#
# 命名协议（必须遵守）：
#   图像文件名格式：{seqID}_{frameID}.jpg
#   示例：seq04_00594.jpg
#   frameID 为视频原始帧号（绝对值），非等间隔抽帧时保持原始帧号

# ─────────────────────────────────────────────────────────
# 0. 自定义模块注册（必须置于最顶部）
# ─────────────────────────────────────────────────────────
custom_imports = dict(
    imports=[
        'mmrotate.models',                             # 原 baseline
        'crane_project.mmrotate_crane.sym_eood.sym_eood_detector',
        'crane_project.mmrotate_crane.sym_eood.sym_eood_head',
        'crane_project.mmrotate_crane.sym_eood.sym_kld_loss',
        'crane_project.mmrotate_crane.crane_metrics'
    ],
    allow_failed_imports=False
)

# ─────────────────────────────────────────────────────────
# 1. 继承 MMRotate 官方 base 配置
# ─────────────────────────────────────────────────────────
_base_ = [
    '../_base_/datasetes/crane_dota.py',  # 完全相同的继承
    'mmrotate::_base_/models/retinanet_obb_r50_fpn.py',
    'mmrotate::_base_/schedules/schedule_1x.py',
    'mmrotate::_base_/default_runtime.py',
]

# ─────────────────────────────────────────────────────────
# 2. 模型配置（数据集、dataloader、evaluator 均继承 base）
#    EOOD baseline：RetinaNet-O
# ─────────────────────────────────────────────────────────
model = dict(
    type='SymEOOD',  # [核心创新] 我们挂载的拓扑隔离单阶段检测器基座
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),

    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
        ),
    ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
    ),

    bbox_head=dict(

        type='SymEOODHead',   # [核心创新] 拦截特征图，注入物理框的改造头
        num_classes=1,        # 明确单目标：门座式起重机抓斗
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            # [工程调优] 抓斗通常具有极端长宽比，移除接近 1 的正方形框，增加长条状先验
            ratios=[0.2, 0.5, 2.0, 5.0], 
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        loss_cls=dict(
            type='SymNFLLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            # 【极其关键】：这里的 tau_init, tau_min 和 warmup_iters 
            # 必须与 train_cfg 中 assigner 的配置保持绝对相同的数值！
            tau_init=10.0,      
            tau_min=1.0,        
            warmup_iters=500,   # 与 assigner / LinearLR 保持一致
            ),
        loss_bbox=dict(
            type='SymKLDLoss',      # 调用你落盘的几何回归损失
            eps=1e-6,
            reduction='mean',
            loss_weight=2.0         # 适当放大回归权重以强调高精度对齐
            ),
    ),
    #辅助头配置
    aux_bbox_head=[dict(
        type='RotatedATSSHead',  # 注意前缀 Rotated
        num_classes=1,           # 单目标抓斗
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='FakeRotatedBBox',
            angle_version='le90',
            scale_factor=8.0),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='RotatedIoULoss', # 直接调用已有的模块
            mode='linear',
            loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)
    )],
    
    train_cfg=dict(
        assigner=dict(
            type='SymPOLAAssigner',
            cost_class=1.0,
            cost_reg=1.0,
            o2m=False,        # 关闭一对多，开启强制 Top-1 极值搜索
            tau_init=10.0,    # 预热期高温系数
            tau_min=1.0,      # 衰减后的基准温度
            warmup_iters=500,  # 与 loss_cls / LinearLR 保持一致
        ),
        # 辅助头：维持一对多密集分配
        aux_assigner=[
            dict(
                type='RotatedATSSAssigner',
                topk=9
            )
        ],
        
        # 必须关闭 NMS 算子以实现端到端，或将 NMS 阈值设为极高（如 0.99）以架空其作用
        allowed_border=-1,
        pos_weight=-1,
        debug=False 
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=1,             # 单目标：最多输出 1 个框
    ),
)

# ─────────────────────────────────────────────────────────
# 3. 训练调度
# ─────────────────────────────────────────────────────────
max_epochs = 24

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2,
)
val_cfg  = dict(type='ValLoop')
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
# 4. 运行时配置
# ─────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=5,
        save_best='crane/sim/ACI',
        # 以 ACI 为最优模型保存标准：
        #   ACI 越高代表时序越平稳，控制系统越友好
        # 若要以角度精度为标准，改为：
        #   save_best='crane/A-RMSE(deg)', rule='less'
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'),
)

visualizer = dict(
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    name='visualizer',
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level    = 'INFO'
load_from    = None
resume       = False
work_dir     = 'work_dirs/crane_symeood'
