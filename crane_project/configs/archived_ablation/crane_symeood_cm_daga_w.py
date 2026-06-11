# crane_symeood_cm_daga_w.py
#  测试效果是否可行
# 效果一般
# CM-DAGA-W: M2 + 主头置信度调制辅助回归权重
#   - 保留 M2 的 ATSS 分配 (ATSSObbAssigner) 不变
#   - 保留 M2 的 FocalLoss 分类不变
#   - 保留 M2 的 SmoothL1 回归不变
#   - 仅在辅助头 bbox loss 上叠加 GT-level 主头置信度权重 w_cm
#       w_cm = cm_min_weight + (1 - cm_min_weight) * (1 - s_main_gt)
#       w_cm 用 detach 的主头 cls_score 估计，warmup 期间线性放大
#   - 不引入 SymKLD / DAGA difficulty / threshold modulation / weak positive expansion

_base_ = ['crane_symeood_m2.py']

angle_version = 'le90'

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.cm_daga_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

model = dict(
    aux_bbox_head=[dict(
        type='CMDAGAHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,

        # CM-DAGA-W 专属参数
        cm_min_weight=0.5,        # 高置信 GT 的辅助回归最低保留比例
        cm_warmup_iters=1000,     # 与主头 warmup 对齐
        cm_modulate_cls=False,    # v1 不削弱辅助分类（保留召回监督）
        cm_modulate_bbox=True,    # v1 仅调制辅助回归 loss
        cm_topk=9,                # 估计 GT 主头置信度时取最近的 K 个 anchor

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

# 每 2 个 epoch 保存一次；24 epoch 训练结束后只保留后 5 个：
# epoch_16 / epoch_18 / epoch_20 / epoch_22 / epoch_24
checkpoint_config = dict(interval=2, max_keep_ckpts=5)

work_dir = 'work_dirs/crane_symeood_cm_daga_w'
