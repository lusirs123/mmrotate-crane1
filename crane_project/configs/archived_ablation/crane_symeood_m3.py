# 性能还是不佳，放弃
# crane_symeood_m3.py
# 消融实验 M3：SymEOOD 主头 + RotatedATSS 辅助头 + UADH
#
# 设计目的：
#   - 继承 M2 的完整配置，保留 RotatedATSSHead auxiliary head；
#   - 额外加入 UADHead，作为不确定性感知的 diagonal-scale consistency 正则；
#   - 验证 UADH 是否能在不移除 ATSS 完整 OBB/角度辅助监督的前提下，进一步提升时序稳定性与控制适用性。
#
# 与已有实验的关系：
#   M1: SymEOOD 主头，无辅助头
#   M2: SymEOOD 主头 + RotatedATSSHead auxiliary head
#   UAHD: SymEOOD 主头 + UADHead，关闭 RotatedATSSHead
#   M3: SymEOOD 主头 + RotatedATSSHead auxiliary head + UADHead

# 注意：
#   当前 UADHead 监督的是 diagonal-scale / uncertainty，不提供显式角度回归监督；
#   因此 M3 中保留 RotatedATSSHead，用其提供完整 DeltaXYWHA 辅助监督，
#   UADHead 只作为轻量正则，不替代 ATSS auxiliary head。

_base_ = ['crane_symeood_m2.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.uadh_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

# 继承 M2 的全部模型结构与训练参数。
# 这里不覆写 aux_bbox_head，因此会保留 M2 中的 RotatedATSSHead。
# 仅额外加入 UADHead。由于 M2 已经有 ATSS 辅助损失，UADH 权重设置得更保守，
# 避免 diagonal-scale 正则干扰主头和 ATSS 的完整 OBB/角度监督。
model = dict(
    uadh_head=dict(
        type='UADHead',
        in_channels=256,
        feat_channels=128,
        mid_channels=64,
        loss_weight_nll=0.02,
        loss_weight_consistency=0.005,
        gaussian_sigma_ratio=0.25,
        mask_threshold=0.1,
        min_log_var=-6.0,
        max_log_var=4.0,
    ),
)

work_dir = 'work_dirs/crane_symeood_m3'
