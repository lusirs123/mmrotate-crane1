# =========================================================
# SymEOOD(K=1) + Score-Level Context Modulation + Brightness Aug
#
# 核心改动（对比旧 auxctx / k1_brightaug）：
#   架构转向: feature-level injection → score-level modulation
#   1. 外扩头不再是"特征增强器"，而是"分类分数空间偏置"
#      - 旧: context logit → tanh gate → FPN feature * (1+scale*gate)
#      - 新: context logit → sigmoid(detach) → cls_logit += gate_alpha * map
#   2. FPN 特征完全不被调制 → 几何分支（reg/angle）不受影响
#   3. gate_alpha 是可学习标量（nn.Parameter），通过 cls loss 优化
#   4. context head 训练完全独立（BCE+Dice loss，FPN detach）
#   5. 推理期 context head 也运行，输出 platform map + gate_alpha 施加 score boost
#
# 设计动机：
#   Probe P1-P5 证明: baseline geometry 已经可用（decoded-neigh 0.620）
#   问题是 100% scoring（global_max ≈ 0.006）
#   Feature modulation 会摧毁几何（0.620→0.143）
#   所以只动 score，不动 feature
#
# 守门线：sim A-RMSE ≤ 1.45 / real 全段 A-RMSE 不退化
# 期望：real_seq02 死段 global_max 提升，MCML 下降
# =========================================================

_base_ = ['./crane_symeood_k1_brightaug.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.platform_context_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

# 平台标定 K 值（与之前版本一致）
seq_platform_k = dict(
    real_seq01=dict(
        width_k=0.98732591,
        height_k=1.62260842,
        offset_long_k=0.00934639,
        offset_short_k=0.00164862,
        dtheta=0.0),
    real_seq04=dict(
        width_k=0.77059758,
        height_k=1.53177023,
        offset_long_k=0.01926015,
        offset_short_k=0.03946324,
        dtheta=0.0),
    real_seq05=dict(
        width_k=0.99586868,
        height_k=1.62817121,
        offset_long_k=0.02955090,
        offset_short_k=-0.02368786,
        dtheta=0.0),
    real_seq06=dict(
        width_k=0.76033056,
        height_k=1.49701762,
        offset_long_k=0.00316936,
        offset_short_k=0.02095963,
        dtheta=0.0),
    sim_seq08=dict(
        width_k=0.69372499,
        height_k=1.74030745,
        offset_long_k=0.01328498,
        offset_short_k=0.03850464,
        dtheta=0.0),
)

model = dict(
    # ============================================
    # Bbox head: 添加 score-level context modulation
    # ============================================
    bbox_head=dict(
        # Score-level context modulation with sigmoid gate
        # effective_gate = gate_scale * σ(gate_alpha) ∈ [0, gate_scale]
        # 上轮教训: gate_alpha 无约束膨胀到 0.186 → VAL A-RMSE 14-21° 崩溃
        # 此次修复: sigmoid 门控 + gate_scale=0.05 硬上限
        use_score_context_modulation=True,
        score_context_gate_init=0.0,
        score_context_gate_scale=0.05,
    ),

    # ============================================
    # Platform context head: 训练期独立预测平台区域
    # ============================================
    platform_context_head=dict(
        type='PlatformContextHead',
        in_channels=256,
        feat_channels=128,
        stacked_convs=2,
        levels=(0, 1, 2),      # 只在 P3-P5 上预测
        seq_platform_k=seq_platform_k,
        loss_weight=0.1,        # 比 original(0.05) 高，context 质量直接影响 score boost
        pos_weight=5.0,
        neg_weight=0.02,
        min_pos_pixels=1,
        use_dtheta=False,
        # Dice 损失：处理极端类别不平衡（平台 <1% 像素）
        use_dice_loss=True,
        dice_weight=1.0,
        # 目标膨胀：扩大正样本区域
        target_dilate_k=1.5,
    ),

    # 移除 feature-level injection
    platform_context_injector=None,
    inject_aux_only=False,
)

work_dir = 'work_dirs/crane_symeood_k1_auxctx'
