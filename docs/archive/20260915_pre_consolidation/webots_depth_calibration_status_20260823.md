# Webots 单目尺度标定状态（2026-08-23）

## 证据边界

- 全流程不使用 PLC；Webots GPS 仅用于仿真真值交叉审计，不是模型输入。
- `calibration_train` 只用于拟合或训练侧机制审计。
- `fixed_dev` 、未知序列泛化与真实部署必须单独报告，不与标定结果混用。
- 真实视频没有米制深度真值，不报告真实域绝对深度精度。

## calibration_train_01：连续垂向标定

- 帧数：550。
- 作用：拟合 `Z_CG,opt = k (l'_diag)^alpha`。
- 拟合参数：`k=1764.5919116475932`，`alpha=-1.0005117394222605`。
- 训练内误差：MAE 5.847 mm，RMSE 7.056 mm，AbsRel 0.0354%，最大绝对误差 14.690 mm。

## calibration_train_02_yaw_small_step：小幅自旋机制审计

- schema：`webots_crane_depth_gt_v7`。
- 帧数：1143；图像、标签和 JSONL 全部对齐，帧序号连续。
- 采样：12.5 FPS，相邻帧固定间隔 0.08 s，单一 run ID。
- 有效性：1143/1143 OBB 有效、深度真值有效、运行时真值审计通过。
- 三个完整自旋循环分别位于约 8.3 m、16.4 m 和 25.5 m。
- 相对机械正位自旋角完整覆盖 `[-8 deg, +8 deg]`。
- 总摆角范围：0.005 deg--2.514 deg；仅 38 帧大于 2 deg，没有帧大于 3 deg。
- 直接使用 `calibration_train_01` 参数时：MAE 7.331 mm，RMSE 8.562 mm，AbsRel 0.0542%，最大绝对误差 19.503 mm。
- 绝对残差与绝对自旋角的相关系数为 0.0366，与总摆角的相关系数为 0.4617。
- 裁决：小幅自旋不是当前尺度误差的主要来源；摆角是更值得在 `fixed_dev` 中独立验证的因素。
- 本序列是训练侧机制审计，不由于帧数更多而重新加权拟合 `k, alpha`。

## 已冻结的暂定标定文件

- `crane_project/configs/depth_scale_calibration_provisional_v1.json`
- 只使用 `calibration_train_01` 拟合参数。
- `calibration_train_02_yaw_small_step` 只提供训练侧自旋应力审计。
- 在固定 `fixed_dev` 运行前不再改参数。

## Pilot 数据处理

- `depth_pilot_04` 和 `depth_pilot_05_compact` 仅用于完整真值审计与 compact JSONL 存储流程调试。
- 它们不进入标定拟合、`fixed_dev`、未知序列泛化或部署证据。
- 正式 v6/v7 序列已通过相同运行时真值审计，因此 pilot 原始图像和标签可删除；其阶段性结论保留在本文档中。

## 唯一下一步

采集一条预先固定的 `fixed_dev_01_unseen_depth_swing`，使用未在标定中重复的连续深度轨迹与自然摆动，不开启主动自旋。该序列只验证已冻结的 `k, alpha`，不回填拟合。
