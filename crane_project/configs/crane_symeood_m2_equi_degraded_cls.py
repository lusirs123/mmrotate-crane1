# crane_project/configs/crane_symeood_m2_equi_degraded_cls.py
#
# M2 + L_equi + degraded cls-only branch.
#
# 目的:
#   验证失败消融后的核心假设 H1:
#     L_equi + SimpleAug 的冲突主要来自低 SNR 退化图上的 bbox/angle/equi
#     几何梯度。这里保持 clean branch 完全沿用 M2+L_equi，只新增一条
#     degraded branch，并且该分支只计算分类/objectness loss。
#    它证明“训练端低质图直接参与几何会冲突，cls-only 解耦更健康”，但它自己不是最优解。
# 损失结构:
#   clean branch:
#     L_cls + L_bbox + L_equi
#   degraded branch:
#     loss_degraded_cls only
#     no bbox / no angle / no L_equi
#
# 注意:
#   本配置暂不引入线性光空间变量，degraded view 复刻 SimpleAug 的
#   gamma/sRGB 风格。若该实验成立，再单独做线性光空间消融。

_base_ = ['./crane_symeood_m2_equi.py']

model = dict(
    bbox_head=dict(
        use_degraded_cls_loss=True,
        degraded_cls_loss_weight=0.25,
        degraded_brightness_range=(0.4, 1.0),
        degraded_contrast_range=(0.6, 1.2),
        degraded_noise_std_range=(0.0, 20.0),
        degraded_prob=0.5,
    )
)

work_dir = 'work_dirs/crane_symeood_m2_equi_degraded_cls'
