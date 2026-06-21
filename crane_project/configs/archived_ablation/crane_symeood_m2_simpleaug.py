# ============================================================
# 实验配置: M2 + 简单随机数据增强 (Step 1)
# ============================================================
#
# 【实验目的】
# 验证基础的亮度/对比度/噪声数据增强能否降低 test real 域的 MCML。
# 这是 HRAA 方案的第一步：先用最简单的随机增强确认"增强本身
# 是否有效"，再决定是否需要 failure-driven 的精细增强。
#
# 【与 M2 的唯一差异】
# 在 train pipeline 中新增 RandomBrightnessContrast 增强算子，
# 置于 RRandomFlip 之后、Normalize 之前。
# 模型结构、损失函数、学习率、epoch 数等全部与 crane_symeood_m2 一致。
#
# 【增强设计思路】
# test real 域的失败帧主要是夜间/低照度场景（failure audit 已确认），
# 因此 brightness_range 偏向降暗（gamma 0.4~1.0，只降不升），
# 用以模拟低光照条件；contrast_range 覆盖低对比（雾天/粉尘）
# 到正常对比；noise_std_range 模拟传感器噪声。
#
# 【消融对照关系】
#   Baseline (Rotated RetinaNet)  → 原始 baseline
#   M2 (crane_symeood_m2)        → 当前最优模型，MCML_max=44
#   M2 + 简单增强 (本配置)        → 验证增强是否有收益
#   M2 + HRAA (后续)             → failure-driven 精细增强
#
# ============================================================

# ---------- 评估配置 ----------
# 每 2 个 epoch 在 val 集上评估一次 mAP
# save_best: 以加权中心召回率 (Weighted_R_center) 为模型选择基准
# thresh_sim/real: 中心偏移容差阈值（像素），用于计算 R_center
# weight_sim/real: sim 域和 real 域的加权比例 (0.7:0.3)
evaluation = dict(
    interval=2,
    metric='mAP',
    save_best='Weighted_R_center',
    rule='greater',
    thresh_sim=10.0,
    thresh_real=25.0,
    weight_sim=0.7,
    weight_real=0.3)

# ---------- 优化器配置 ----------
# SGD 优化器，lr=0.0025，动量 0.9，权重衰减 0.0001
# 梯度裁剪: L2 范数上限 10，防止梯度爆炸
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# ---------- 学习率调度 ----------
# StepLR: 在 epoch 16 和 22 衰减学习率
# 线性 warmup: 前 1000 次迭代从 lr*0.001 线性增长到 lr
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 22])

# ---------- 训练器配置 ----------
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# 不从断点恢复，从头训练（确保增强从第一个 epoch 就生效）
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

# ---------- 自定义模块导入 ----------
# SymEOOD 的核心自定义组件:
#   - crane_custom_dota: 自定义 CraneDataset 数据集类
#   - sym_eood_detector: SymEOOD 检测器
#   - sym_eood_head: SymEOOD 检测头
#   - sym_nfl_loss: SymNFL 负样本焦点损失（含温度调度）
#   - sym_kld_loss: SymKLD 各向异性对称 KL 散度损失
#   - sym_pola: SymPOLA 物理约束角度感知标签分配器
custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola'
    ],
    allow_failed_imports=False)

angle_version = 'le90'
max_epochs = 24

# ============================================================
# 模型配置（与 M2 完全一致）
# ============================================================
#
# 【整体架构】SymEOOD = ResNet50-FPN + SymEOODHead + RotatedATSS 辅助头
#   - 主头 (SymEOODHead): 使用 SymKLD + SymNFL + SymPOLA，端到端无 NMS
#   - 辅助头 (RotatedATSSHead): 一对多密集监督，推理时物理熔断
#
# 【与 Baseline 的区别】
#   Baseline 使用 Rotated RetinaNet + MaxIoU 分配 + NMS
#   M2 使用 SymEOODHead + SymPOLA 一对一匹配 + 辅助头
#
model = dict(
    type='SymEOOD',
    # --- Backbone: ResNet-50，预训练权重来自 torchvision ---
    # frozen_stages=1: 冻结 stem + 第一层，只微调后三层
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
    # --- Neck: FPN，5 级特征融合 ---
    # in_channels: ResNet-50 各层输出通道 [256, 512, 1024, 2048]
    # out_channels: 统一到 256
    # start_level=1: 从 C2 开始（C1 分辨率太大，不参与）
    # add_extra_convs='on_input': 在输入特征图上加额外卷积生成 P6
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    # --- 主检测头: SymEOODHead ---
    # 4 层卷积，256 通道，3 种 anchor 比例 [0.5, 1.0, 2.0]
    # 损失函数: SymNFL (分类) + SymKLD (回归)
    # 标签分配: SymPOLA (一对一极值匹配)
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
            angle_range='le90',
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
                type='Normal', name='retina_cls', std=0.01, bias_prob=0.01)),
        # SymNFL: 负样本重加权焦点损失 + 温度调度
        # tau_init=10.0 → tau_min=1.0: 训练初期放松角度惩罚，逐步收紧
        # loss_weight=0.25: 分类损失权重
        loss_cls=dict(
            type='SymNFLLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            tau_init=10.0,
            tau_min=1.0,
            warmup_iters=1000,
            loss_call_factor=5,
            eps=1e-06,
            reduction='mean',
            loss_weight=0.25,
            kld_chunk_size=256),
        # SymKLD: 各向异性对称 KL 散度损失
        # 将 OBB 建模为 2D 高斯分布，用精度矩阵对角度施加指数级惩罚
        # loss_weight=2.0: 回归损失权重
        loss_bbox=dict(
            type='SymKLDLoss', eps=1e-06, reduction='mean', loss_weight=2.0)),
    # --- 训练配置: SymPOLA 标签分配 ---
    # 一对一极值匹配: cost_class=1.0, cost_reg=2.0
    # o2m=False: 单头模式（辅助头单独处理一对多）
    # topk=1: 每个 GT 只匹配 1 个预测
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
            eps=1e-06),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    # --- 测试配置 ---
    # score_thr=0.05: 置信度阈值（较低，依赖 NMS 过滤）
    # nms iou_thr=0.1: 极低 IoU 阈值，因为 SymEOOD 理论上输出单峰分布
    # max_per_img=1: 单目标场景，每张图最多保留 1 个检测
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=1),
    # --- 辅助头: RotatedATSSHead（一对多密集监督）---
    # 功能: 训练期提供密集梯度信号，缓解一对一匹配的监督稀疏问题
    # 推理期: 物理熔断，不参与最终输出，零额外算力开销
    # anchor 只用 1 种比例 [1.0]，与主头的 [0.5, 1.0, 2.0] 互补
    # 损失权重极低 (0.01): 仅做辅助监督，不干扰主头优化
    aux_bbox_head=[
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
                angle_range='le90',
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
                    type='Normal', name='retina_cls', std=0.01,
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
                    angle_version='le90',
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                nms_pre=2000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(iou_thr=0.1),
                max_per_img=1))
    ])

# ============================================================
# 数据集配置
# ============================================================

dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'

# ---------- 训练数据流 ----------
# 包含两个子集:
#   1. train/annfiles/  — 真实港口数据（约 2600 帧，7 条序列）
#   2. train_sim/annfiles/ — Webots 仿真数据（约 1300 帧，2 条序列）
# 两个子集使用相同的 train_pipeline
#
# 流水线顺序:
#   LoadImageFromFile → LoadAnnotations → RResize → RRandomFlip
#   → RandomBrightnessContrast (新增!) → Normalize → Pad → Format → Collect
#
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 缩放到 1024×1024，保持长宽比
    dict(type='RResize', img_scale=(1024, 1024)),
    # 随机翻转: 水平/垂直/对角线各 25% 概率
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    # >>> 新增: 简单亮度/对比度/噪声数据增强 <<<
    # 目的: 模拟 test real 域中的夜间/低照度/高噪声场景
    #
    # brightness_range=(0.4, 1.0):
    #   使用 gamma 校正，gamma < 1 使图像变暗。
    #   0.4 ≈ 夜间低光照，1.0 = 不改变。
    #   只降不升，因为 test 失败帧主要是低光场景。
    #
    # contrast_range=(0.6, 1.2):
    #   < 1 降低对比度（模拟雾天/粉尘散射），
    #   > 1 增强对比度（模拟局部强光）。
    #
    # noise_std_range=(0, 20):
    #   高斯噪声标准差，0-255 尺度。
    #   0 = 无噪声，20 = 中等噪声（模拟传感器热噪声）。
    #
    # prob=0.5:
    #   50% 的帧会被增强，保留一半原始分布。
    #   避免增强过强导致模型忘记正常场景的特征。
    #
    dict(
        type='RandomBrightnessContrast',
        brightness_range=(0.4, 1.0),
        contrast_range=(0.6, 1.2),
        noise_std_range=(0, 20),
        prob=0.5),
    # ImageNet 标准归一化（在增强之后，确保增强效果不被归一化抵消）
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    # 填充到 1024×1024，填充值 114（灰色）
    dict(
        type='Pad', size=(1024, 1024),
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# ---------- 测试/验证数据流 ----------
# 不做任何增强，只做缩放 + 归一化 + 填充
# 用于 val 集评估和 test 集最终测试
test_pipeline = [
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
                type='Pad',
                size=(1024, 1024),
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

# ---------- 数据加载配置 ----------
# samples_per_gpu=2: 每张 GPU 每次取 2 个样本
# workers_per_gpu=2: 每张 GPU 2 个数据加载子进程
# 训练集: real + sim 混合训练（ConcatDataset）
# 验证集: val 集，用于训练期间的在线监控和离线权重选择
# 测试集: test 集，仅在最终权重选出后评估一次
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type='CraneDataset',
            data_root='crane_project/data/crane_grab/',
            ann_file='train/annfiles/',
            img_prefix='train/images/',
            pipeline=train_pipeline,
            version='le90'),
        dict(
            type='CraneDataset',
            data_root='crane_project/data/crane_grab/',
            ann_file='train_sim/annfiles/',
            img_prefix='train/images/',
            pipeline=train_pipeline,
            version='le90')
    ],
    val=dict(
        type='CraneDataset',
        data_root='crane_project/data/crane_grab/',
        ann_file='val/annfiles/',
        img_prefix='val/images/',
        pipeline=test_pipeline,
        version='le90'),
    test=dict(
        type='CraneDataset',
        data_root='crane_project/data/crane_grab/',
        ann_file='test/annfiles/',
        img_prefix='test/images/',
        pipeline=test_pipeline,
        version='le90'))

# ============================================================
# 输出与运行配置
# ============================================================

# work_dir: 训练日志和 checkpoint 的保存目录
work_dir = 'work_dirs/crane_symeood_m2_simpleaug'
auto_resume = False
# 使用 2 张 GPU 训练
gpu_ids = range(0, 2)
