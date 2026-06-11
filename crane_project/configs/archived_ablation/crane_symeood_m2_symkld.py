# crane_symeood_m2_symkld.py
# =========================================================
# M2 + weak SymKLD add-on（v1，λ_kld = 0.1）
# =========================================================
#
# 【本版修改内容】
#
# 1. 新建辅助头类型 RotatedATSSSymKLDHead
#    - 文件：mmrotate/models/dense_heads/rotated_atss_symkld_head.py
#    - 继承 RotatedATSSHead，重写 loss_single()
#    - 保留 M2 原有的 SmoothL1 delta 回归路径（完全不变）
#    - 额外对正样本 decode 到物理 OBB，计算 SymKLD，加到 bbox loss 上
#    - 不使用 anisotropic Gaussian weighting，不替换基础监督
# 3. 本配置文件
#    - 继承 crane_symeood_m2.py，只替换辅助头类型和新增 loss_kld
#    - 主头（SymEOODHead + SymPOLA）、训练策略、数据流形完全不变
# 【与已有实验的关系】
#   M1:        SymEOOD 主头，无辅助头
#   M2:        SymEOOD 主头 + RotatedATSSHead 辅助头（SmoothL1）
#   UAHD:      SymEOOD 主头 + UADHead
#   M3:        SymEOOD 主头 + RotatedATSSHead + UADHead
#   AGAS:      SymEOOD 主头 + AGASHead 辅助头（SymKLD + anisotropic Gaussian）
#   M2_SymKLD: SymEOOD 主头 + RotatedATSSSymKLDHead 辅助头（SmoothL1 + weak SymKLD）
#
# 【核心假设】
#   AGAS 实验表明：
#     - SymKLD 辅助梯度过强会损害 real 域召回（TDR 从 77 降到 60/75）
#     - 但 sim/A-RMSE 保持优势，说明 SymKLD 对角度/宽高比几何确实有帮助
#   因此本实验保留 M2 的宽泛 SmoothL1 监护，只将 SymKLD 作为轻量几何增强项。
#
# 【论文叙事】
#   M2 证明宽泛 ATSS 辅助监督有效；
#   AGAS 证明强几何替换会损害 real 域；
#   因此我们在 M2 基础上加入轻量 SymKLD 几何一致项，
#   既保留召回稳定性，又引入 OBB 分布度量。
#
# 【预期结果】
#   成功：real/TDR_w10 >= 77.11, real/MCML_max <= 44, sim/A-RMSE 可能略优于 1.5832
#   失败：说明 SymKLD 即使很弱也会引入不稳定 -> 转向 priority 3 gating

_base_ = ['crane_symeood_m2.py']

angle_version = 'le90'

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.rotated_atss_symkld_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

model = dict(
    aux_bbox_head=[dict(
        type='RotatedATSSSymKLDHead',   # 从 RotatedATSSHead 改为 RotatedATSSSymKLDHead
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
        # 分类沿用 M2 的 FocalLoss，保证辅助分类分支稳定（不变）
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.01),
        # 回归保持 M2 的 SmoothL1 delta 回归（不变）
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.01),
        # 【新增】轻量 SymKLD 几何一致项
        # lambda_kld = 0.001 -> 相对于 SmoothL1 (0.01) 的比例约为 0.1
        # 后续可尝试 0.002 (比例 0.2)
        loss_kld=dict(
            type='SymKLDLoss',
            eps=1e-6,
            reduction='mean',
            loss_weight=0.001),
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

work_dir = 'work_dirs/crane_symeood_m2_symkld_v1'
