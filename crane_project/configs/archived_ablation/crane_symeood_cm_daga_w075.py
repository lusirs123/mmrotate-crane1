# crane_symeood_cm_daga_w075.py
# CM-DAGA-W-075: 在 CM-DAGA-W 基础上减弱辅助回归降权幅度
#   - 仅将 cm_min_weight 从 0.5 提高到 0.75
#   - 目的：验证 CM-DAGA-W real/MCML 变差是否来自辅助 bbox 回归被削得过轻
#   - 其余 ATSS 分配、FocalLoss、SmoothL1、训练策略、checkpoint 策略保持不变

_base_ = ['crane_symeood_cm_daga_w.py']

angle_version = 'le90'

model = dict(
    aux_bbox_head=[dict(
        type='CMDAGAHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,

        # CM-DAGA-W-075: 更保守地减压，保留更多 M2 辅助回归支撑
        cm_min_weight=0.75,
        cm_warmup_iters=1000,
        cm_modulate_cls=False,
        cm_modulate_bbox=True,
        cm_topk=9,

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
)

work_dir = 'work_dirs/crane_symeood_cm_daga_w075'
