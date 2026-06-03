#性能不佳，放弃，修改为AGAS
# crane_symeood_UAHD.py
# 消融实验：SymEOOD + UADH（替换 RotatedATSS 辅助头）= M2 + UAHD
# 继承 M2 全部参数，仅覆写辅助头：
#   - aux_bbox_head=None（关闭 RotatedATSS）
#   - uadh_head=dict(type='UADHead', ...)（启用 UADH）
#   - custom_imports 新增 uadh_head 模块
# 其余超参（backbone/neck/head/数据/优化器/调度/评估）全部继承 M2，零人工对齐
# 但是这里为各项同性并且没有实现度量同构

_base_ = ['crane_symeood_m2.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.uadh_head',       # ← 新增
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola'
    ],
    allow_failed_imports=False)

# 深度合并：仅覆写变化的字段，其余继承 M2
model = dict(
    aux_bbox_head=None,       # 关闭 RotatedATSS 辅助头
    uadh_head=dict(           # 启用 UADH 辅助头
        type='UADHead',
        in_channels=256,
        feat_channels=128,
        mid_channels=64,
        loss_weight_nll=0.05,
        loss_weight_consistency=0.01,
        gaussian_sigma_ratio=0.25,
        mask_threshold=0.1,
        min_log_var=-6.0,
        max_log_var=4.0,
    ),
)

work_dir = 'work_dirs/crane_symeood_UAHD'
