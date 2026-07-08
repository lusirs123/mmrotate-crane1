# crane_project/configs/crane_symeood_m2_equi_aux2_sgldet.py
#
# M2 + L_equi + aux2 退化视图前景鲁棒分支。
#
# 整体分工:
#   主头:
#     clean image -> SymEOODHead
#     L_cls + L_bbox + L_equi
#     负责最终推理、角度/几何精度和 NMS-Free 单峰输出。
#
#   aux1:
#     clean image -> 普通 RotatedATSSHead
#     沿用 M2 的一对多密集监督, 负责冷启动和稠密梯度。
#
#   aux2:
#     degraded image -> DegradedATSSForegroundHead
#     使用 ATSS 一对多分配, 只计算前景/objectness 分类损失;
#     bbox/angle loss 恒为 0, 不让退化视图污染 OBB 几何。
#
# 关键纪律:
#   1. 退化视图只作为 OOD/低光召回辅助, 不作为几何监督来源。
#   2. aux2 的幅度域一致性使用 clean feature stop-grad teacher,
#      只把退化特征拉向 clean 幅度谱, 不约束相位。
#   3. aux1/aux2 推理期全部熔断, 推理仍只走主头, 零额外开销。
#   4. 这是对 degraded-cls 的结构化升级:
#      从“退化图复用主头分类塔”改为“退化图使用独立 ATSS cls-only 辅助头”。
#
# 判读优先级:
#   第一守门:
#     sim/A-RMSE、real/R_center、real/ACI 不能明显劣化。
#   第二收益:
#     real/TDR_w10、real/MCML_mean、hard-real DEAD-global rate 是否改善。
#   若 nominal 几何变差, 说明 aux2 共享 backbone 的残余耦合仍过强,
#   应降低 aux2 loss_cls.loss_weight / degraded_aux2_amp_loss_weight。

_base_ = ['./crane_symeood_m2_equi.py']

angle_version = 'le90'

model = dict(
    bbox_head=dict(
        # 保留主头 L_equi; 旧 degraded-cls 关闭, 避免退化图更新主头分类塔。
        use_equi_loss=True,
        equi_loss_weight=0.2,
        use_degraded_cls_loss=False,
        use_degraded_aux2_loss=False,

        # 物理先验退化: 只改光照/颜色/噪声, 不改 OBB 几何。
        # 这些范围不使用 test 统计反推; test OOD 指标只用于事后核对包络性。
        degraded_prob=0.6,
        degraded_brightness_range=(0.35, 0.95),
        degraded_contrast_range=(0.55, 1.25),
        degraded_noise_std_range=(0.0, 18.0),

        # 上亮下暗的垂直照度场。ratio > 1 表示上侧更亮、下侧更暗。
        degraded_vertical_grad_range=(1.0, 2.4),

        # RGB 通道偏色。输入是 RGB; R/G < 1 对应绿偏, B/G 给少量冷/暖漂移。
        degraded_rg_range=(0.65, 1.05),
        degraded_bg_range=(0.75, 1.15),
    ),

    # 两个辅助头一头一职:
    #   aux0 = clean 普通 ATSS, 补一对一监督稀疏;
    #   aux1 = degraded ATSS cls-only, 补低光/OOD 前景召回。
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
            degraded_aux2_amp_loss_weight=0.03,
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
            # aux2 权重要低于主头, 只作为前景召回正则。
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.008),
            # 该 loss 会被 DegradedATSSForegroundHead 置零; 保留字段仅为构建兼容。
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

work_dir = 'work_dirs/crane_symeood_m2_equi_aux2_sgldet'
