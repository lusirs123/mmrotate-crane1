# crane_symeood_gauss.py
# 对比实验：SymEOOD + 高斯热图辅助头（替换 RotatedATSS）

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.gaussian_heatmap_head',  # ← 新增
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
    ],
    allow_failed_imports=False)

# 继承基础 symeood config，只覆写模型部分
_base_ = ['crane_symeood.py']

model = dict(
    # 主头完全不变，只替换辅助头
    aux_bbox_head=None,          # 关闭原 RotatedATSS 辅助头
    gaussian_head=dict(          # 启用高斯热图辅助头
        type='GaussianHeatmapHead',
        in_channels=256,
        loss_weight=0.5,
    )
)

work_dir = 'work_dirs/crane_symeood_gauss'