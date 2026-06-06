# crane_symeood_AGAS.py
# 消融实验 AGAS：SymEOOD 主头 + 各向异性高斯加权的 ATSS 辅助头
# 实验结果表明各向异性高斯加权的 ATSS 辅助头产出过强的梯度会影响主头的训练效果，使用更加宽泛更加平和的辅助头
# 设计目的：
#   - 继承 M2 的完整训练/数据/主头配置；
#   - 将 M2 中普通 RotatedATSSHead auxiliary head 替换为 AGASHead；
#   - 保留 ATSSObbAssigner 与 FocalLoss 分类监督，避免破坏 M2 已验证的稳定性；
#   - 将辅助回归监督改为 decoded SymKLD，并对正样本回归损失加入与 GT OBB 对齐的
#     anisotropic Gaussian weighting。

# 与已有实验的关系：
#   M1: SymEOOD 主头，无辅助头
#   M2: SymEOOD 主头 + RotatedATSSHead auxiliary head
#   UAHD: SymEOOD 主头 + diagonal-scale UADHead，关闭 RotatedATSSHead
#   M3: SymEOOD 主头 + RotatedATSSHead auxiliary head + UADHead
#   AGAS: SymEOOD 主头 + AGASHead auxiliary head
#
# 注意：
#   AGAS 不是在 M2 后额外叠加第二个辅助头，而是替换 M2 的 ATSS auxiliary head。
#   这样可以公平验证：在完整 OBB/角度辅助监督框架内，引入 SymKLD 度量一致回归与
#   各向异性旋转高斯正样本加权是否能进一步提升时序/控制稳定性。

_base_ = ['crane_symeood_m2.py']

# 子配置在 MMCV 临时模块中不会直接继承 base 的 Python 变量，
# 因此这里显式写出 angle_version，保证 aux_bbox_head 覆写时可解析。
angle_version = 'le90'

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.agas_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

model = dict(
    aux_bbox_head=[dict(
        type='AGASHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        agas_alpha=2.0,
        agas_beta=0.5,
        agas_min_weight=0.05,
        agas_normalize_weight=False,
        agas_decode_max_size=None,
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
        # AGAS-lite：分类仍沿用 M2 的 FocalLoss，保证辅助分类分支稳定。
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.01),
        # AGAS-lite：回归改为 decoded OBB 上的 SymKLDLoss。
        # AGASHead.loss_single 内部会先 decode pred/target，再施加 anisotropic Gaussian weight。
        # 诊断实验：先将辅助回归权重从 0.01 降到 0.005，验证 AGAS 是否主要是辅助梯度过强。
        loss_bbox=dict(
            type='SymKLDLoss',
            eps=1e-6,
            reduction='mean',
            loss_weight=0.005),
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

work_dir = 'work_dirs/crane_symeood_AGAS_lw0005'
