# crane_project/configs/crane_symeood_degraded_cls_k1.py
#
# SymEOOD(K=1) + degraded cls-only branch.
#
# 目的:
#   在新的 EOOD(K=1) 消融链上重验旧 M2 谱系的 degraded-cls 机制。
#   该实验只验证低质视图上的 cls/objectness 生存监督是否能缓解
#   real seq02 hard slice 的 DEAD-global 问题，不把 L_equi、norm 或
#   dense aux2 混入同一个变量。
#
# 相对 crane_symeood_k1.py:
#   - clean branch 保持不变: L_cls + L_bbox，推理仍只用主头
#   - 新增 degraded branch: loss_degraded_cls only
#   - degraded branch 不计算 bbox / angle / L_equi
#
# 判读重点:
#   1. seq02[129..172] 是否出现 DEAD-global -> EDGE/OK
#   2. subthreshold aux1/dead 是否高于 K1 的 21/44
#   3. TDR_w10 是否保持或小幅提升
#   4. sim/A-RMSE 相对 K1 不明显劣化
#
# 注意:
#   本配置先复用旧 degraded-cls 的 gamma/contrast/noise 范围，保持
#   单变量干净。垂直照度梯度和 RGB 偏色先不打开；若本配置出现
#   aux/objectness 正信号，再单独建立 hard-real signature 版本。

_base_ = ['./crane_symeood_k1.py']

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

work_dir = 'work_dirs/crane_symeood_degraded_cls_k1'
