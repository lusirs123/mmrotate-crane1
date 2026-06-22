# crane_project/configs/crane_symeood_m2_degraded_cls.py
#
# M2 + degraded cls-only branch, without L_equi.
#
# 目的:
#   作为 `crane_symeood_m2_equi_degraded_cls.py` 的反事实消融:
#     - 保留 degraded cls-only 分支
#     - 去掉 L_equi
#   用来回答 degraded-cls 的收益是否依赖 L_equi 几何主线。
#
# 损失结构:
#   clean branch:
#     L_cls + L_bbox
#   degraded branch:
#     loss_degraded_cls only
#     no bbox / no angle / no L_equi
#
# 判读:
#   若该配置的 real/R_center、sim/A-RMSE、real/mean_RIoU 明显弱于
#   `M2 + L_equi + degraded-cls`, 则说明 L_equi 是几何主贡献,
#   degraded-cls 只是外观/cls 生存辅助。

_base_ = ['./crane_symeood_m2.py']

model = dict(
    bbox_head=dict(
        use_equi_loss=False,
        equi_loss_weight=0.0,
        use_degraded_cls_loss=True,
        degraded_cls_loss_weight=0.25,
        degraded_brightness_range=(0.4, 1.0),
        degraded_contrast_range=(0.6, 1.2),
        degraded_noise_std_range=(0.0, 20.0),
        degraded_prob=0.5,
    )
)

work_dir = 'work_dirs/crane_symeood_m2_degraded_cls'
