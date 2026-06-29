# =========================================================
# 消融实验 M1：Unified Gaussian Manifold (3-suite only)
# 学术定位：在 RetinaNet (M0) 基础上，叠加"度量同构三件套"
#         （SymKLDLoss + SymNFLLoss + SymPOLAAssigner），
#          不引入任何辅助头，验证统一高斯流形主框架的独立贡献。
# 关键差异（vs crane_symeood.py / M3 完整版）：
#   - 去除 aux_bbox_head（无 RotatedATSS 辅助监督）
#   - 去除 gaussian_head（无 UADH 不确定性辅助分支）
#   - 仅保留 SymEOODHead 主头 + Sym 三件套
#   - 其余超参严格对齐 crane_symeood.py / crane_baseline.py，确保公平对比
# =========================================================

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
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
# 模型：仅主头（无辅助监督），度量同构三件套全部启用
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
        # ==========================================
        # 创新点 1+2：SymNFLLoss (空间感知对称归一化 Focal)
        # ==========================================
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
        # ==========================================
        # 创新点 1：SymKLDLoss (对称 KL 高斯流形回归)
        # ==========================================
        loss_bbox=dict(
            type='SymKLDLoss',
            eps=1e-6,
            reduction='mean',
            loss_weight=2.0)),
    train_cfg=dict(
        # ==========================================
        # 创新点 3：SymPOLAAssigner (度量同构 O2M→O2O 分配)
        # ==========================================
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

    # ==========================================
    # 【消融关键】：无辅助监督
    # M1 显式不传 aux_bbox_head / gaussian_head
    # detector 内部 self.aux_heads = None / self.gaussian_head = None
    # forward_train 中会自动跳过辅助分支
    # ==========================================
    # aux_bbox_head=None,   ← 不传即为 None
    # gaussian_head=None,   ← 不传即为 None
)

# =========================================================
# 数据流形（与 crane_baseline / crane_symeood 严格对齐）
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
# 训练调度（严格对齐 baseline / symeood，消除调度差异）
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

log_config = dict(interval=50)

log_level = 'INFO'
load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_m1'
