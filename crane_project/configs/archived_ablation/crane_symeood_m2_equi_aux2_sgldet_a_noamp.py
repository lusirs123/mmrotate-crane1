# crane_project/configs/crane_symeood_m2_equi_aux2_sgldet_a_noamp.py
#
# aux2_sgldet 最小消融 A:
#   只关闭 aux2 幅度域一致性, 保留原 aux2 cls 强度和退化概率。
#
# 目的:
#   判断本次负迁移是否主要来自 FPN 幅度谱一致性约束。
#
# 改动:
#   degraded_aux2_amp_loss_weight: 0.03 -> 0.0
#   aux2 loss_cls.loss_weight:     0.008 不变
#   degraded_prob:                 0.6 不变
#
# 守门:
#   sim/A-RMSE <= 1.45 才允许继续讨论 real MCML/TDR。

_base_ = ['./crane_symeood_m2_equi.py']

angle_version = 'le90'

model = dict(
    bbox_head=dict(
        use_equi_loss=True,
        equi_loss_weight=0.2,
        use_degraded_cls_loss=False,
        use_degraded_aux2_loss=False,
        degraded_prob=0.6,
        degraded_brightness_range=(0.35, 0.95),
        degraded_contrast_range=(0.55, 1.25),
        degraded_noise_std_range=(0.0, 18.0),
        degraded_vertical_grad_range=(1.0, 2.4),
        degraded_rg_range=(0.65, 1.05),
        degraded_bg_range=(0.75, 1.15),
    ),
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
        ),
        dict(
            type='DegradedATSSForegroundHead',
            num_classes=1,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            assign_by_circumhbbox=None,
            use_degraded_view=True,
            degraded_aux2_amp_loss_weight=0.0,
            degraded_aux2_amp_levels=(0, 1, 2),
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
                loss_weight=0.008),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.0),
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
        ),
    ],
)

work_dir = 'work_dirs/crane_symeood_m2_equi_aux2_sgldet_a_noamp'

