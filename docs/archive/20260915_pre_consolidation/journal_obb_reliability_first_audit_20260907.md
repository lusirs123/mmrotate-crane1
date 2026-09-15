# 小论文 OBB 观测可靠性首轮审计

日期：2026-09-07。范围限于 `SymEOOD 抓斗旋转检测 -> OBB 分量可靠性判定 -> 连续视觉观测输出`。本文是只读盘点和审计设计，不是模型验收，也不构成可靠性方法有效、未知序列泛化或部署的结论。

## 1. 当前结论

- 新增贡献仍未成立。必须先在相同保留覆盖率下，证明分量可靠性筛选比检测置信度筛选更能减少中心、尺度和周期方向误差。
- 唯一应优先绑定的检测候选，仍是服务器上的 V4+seq11-v2 replay `epoch_10`；按 2026-09-05 交接，它只是 738-frame official source-val 的暂定 source-retention candidate，`source_gate_passed=False`，没有困难块 OOF/CV 支持，不能称为最终模型。
- 本机没有该候选的 checkpoint、冻结合同、source-val 逐帧预测或交接所列的复核包。因此不能在本机对该候选训练或评价可靠性规则，亦不能用旧 K1 输出替代其身份。
- 本机有一套角色明确的旧 K1 source-val 输出，已用于验证审计数据管线可运行。它仅为基线/通路证据，不可和 V4+seq11-v2 指标或图表合并。

## 2. 模型和数据身份清单

| 项目 | 可用状态 | 身份与约束 |
| --- | --- | --- |
| V4+seq11-v2 replay `epoch_10` | 缺失于本机 | 期望路径：`work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407/epoch_10.pth`；训练配置为 `crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay.py`。2984 source-train frames = original 2781 + seq11 train 203；official source-val 738；seq11 aux-val 48 仅同视频机制验证。不得把该候选升格为 source-gated 或最终模型。 |
| V4 replay config | 主工作区可读 | 冻结 Base-V3 teacher、cached DINO proposals、K1 current anchor、4-frame causal history；不读 target/fixed TEST，禁止 domain/sequence/frame routing。config 写明新 checkpoint `source_gate_passed=False`。 |
| Base-V3 teacher | 本轮未本地绑定工件 | replay config 要求一个已 source-promoted epoch9 teacher；其 checkpoint SHA256 从 promotion report 动态读取。此报告不猜测其哈希。 |
| 普通 SymEOOD K1 epoch24 | 本机可读 | config：`crane_project/configs/crane_symeood_k1_source_val_eval.py`；checkpoint：`work_dirs/crane_symeood_k1/epoch_24.pth`；SHA256 `57233e5423de4a9d0a67fd51058cd7d92adaac5d6621f275cc0ec5b0fc7f9ee2`。这是普通 K1，而非 V4+seq11-v2。 |
| K1 source-val predictions | 本机可读 | `work_dirs/crane_symeood_k1/source_val_epoch24_results.pkl`；SHA256 `361ccd8848c162e92477e04e38f87cd7f28209f07129823e84045e8ec74031c9`；738 条、每帧 top prediction 为 `(cx,cy,w,h,angle,score)`，原图像像素、`le90`。 |
| source-val GT | 本机可读 | `crane_project/data/crane_grab/val/annfiles/` 有 738 DOTA polygon txt；对应 738 images；序列为 `real_seq07` 和 `sim_seq10`。标签离线使用，不参与规则设计或推理。 |
| fixed TEST / target-dev / unknown | 本轮未读取 | 不得用作该可靠性规则选择、阈值扫描或统计替代；未知序列仍缺失。 |

## 3. 只读通路统计：K1 source-val（非候选模型结论）

以文件名排序对齐 K1 `results.pkl` 与 738 个 source-val 标注。GT 四点框按代码的 `le90` 约定转换；对有预测的帧，中心误差为像素欧氏距离，尺度误差为 `mean(abs(log(w_hat/w)), abs(log(h_hat/h)))`，方向误差为 pi 周期的最小绝对差。所有缺失预测必须在正式 coverage 中保留，不能静默丢弃。

| 项目 | 结果 |
| --- | --- |
| 对齐 / 检出 / 漏检 | 738 / 736 / 2 |
| 中心误差 px (p50 / p90 / p95 / max) | 1.859 / 5.183 / 6.435 / 13.218 |
| 尺度误差 (p50 / p90 / p95 / max) | 0.0516 / 0.1359 / 0.1584 / 0.2903 |
| 周期方向误差 deg (p50 / p90 / p95 / max) | 1.284 / 4.769 / 6.186 / 10.172 |
| score (p50 / p90 / p95) | 0.9108 / 0.9515 / 0.9547 |
| Pearson(score, 中心/尺度/方向误差) | -0.3247 / -0.2985 / -0.2760 |

这些相关性仅说明该 K1 source-val 上 score 与三种误差存在有限负关联；它们不支持声称“置信度基线已被超越”。中心和方向在该工件中错误尾部很少，正式训练/验证需要按每个分量的错误支持量和 OOF 角色重审，而非从这些数字预设阈值。

## 4. 第一轮可靠性审计协议

1. **冻结输入。** 对一个唯一、可哈希的检测器版本导出全量逐帧表；每行包含 `split_role, sequence, frame, timestamp, prediction_present, cx, cy, w, h, angle_le90, score`，以及不可变 `config_sha256, checkpoint_sha256, prediction_sha256, manifest_sha256`。每帧只保留该模型实际推理的输出，不接管或修正几何。
2. **离线误差标签。** 将 GT 只用于生成中心、尺度、方向、存在性误差标签。方向以 pi 周期表示；近正方形的方向退化必须单独标记或预注册为“不评价方向”，不可伪造角度正确性。漏检计入所有 coverage/risk 曲线。
3. **先审计支持，再拟合。** 在 source-train 的 OOF 预测或职责清晰的独立 calibration split 上，统计 score、归一化尺度、长宽比/近方形标记、边界截断、当前-过去 causal 残差与三类误差的关系。时间特征只能使用当前和过去，时序块边界清除历史；不得按 target 逐帧调规则。
4. **预注册三个输出。** `r_center, r_scale, r_angle` 及各自 `valid`。先比较 score-only、score+current-frame geometry、score+causal residual 三层；整体 valid 必须由已冻结的组合规则导出，不允许评测时挑最好分量。
5. **等 coverage 主比较。** 在每个分量上将 proposed 与 score-only 匹配相同的总体保留率（包括漏检），报告 risk-coverage curve、固定 coverage 的平均/分位误差、错误接受率、正确拒绝率和拒绝代价。不能只报告被保留帧的平均误差。
6. **连续接口后置。** 首先验证逐帧可靠性；随后才比较 `measurement / bounded prediction / unavailable`，报告各状态覆盖、连续 unavailable 长度、延迟。长漏检必须转为 unavailable，历史框不能伪装成新测量。
7. **证据层级。** source OOF/校准用于选规则；official source gate 用于冻结；fixed target-dev 只诊断；未知完整序列验证泛化；资源/时延才支持工程接口结论。五者分表，不混图、不共享阈值选择。

## 5. 服务器精确导出需求

在 `/media/omnisky/personal_files/ljj/symEOOD` 对**已存在**的 V4+seq11-v2 `epoch_10` 做只读打包/导出；不要训练、不要执行 fixed TEST、不要读取未知数据。至少导出：

```text
work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407/
  epoch_10.pth
  checkpoint_eval_summary.json
  geometry_refiner_frozen_contract.json (或实际同等冻结合同)
  source-val epoch_10 results.pkl
  source-val per-frame all-lane prediction JSON
  训练日志中 config 的 resolved copy
  所有上述文件的 SHA256 清单

crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay.py
work_dirs/crane_symeood_dino_conservative_takeover_v2/source_calibration_collect/source_val_fusion_source_audit.json
work_dirs/crane_symeood_dino_distill_support_v1/source_collect/source_train_all_lane_audit.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/
  full_source_contract.json
  blocksplit_v2/audited_split_materialization.json
  blocksplit_v2/train_all_lane_audit.json
  blocksplit_v2/aux_val_all_lane_audit.json
crane_project/data/crane_grab/extra_source_real_seq11_pilot_k1p9_v2/split_manifest.json
```

逐帧 JSON 不能是 `records=0` 的 summary-only `failure_audit_real*.json`。它必须覆盖 official source-val 的全部 738 帧，且包含 filename、split、sequence、frame、timestamp（无则显式 null）、prediction_present、最终 OBB+score、K1/DINO/refiner provenance、原图坐标与 `le90` 角度约定。若需要在服务器生成该 JSON，推理命令必须只针对 official source-val，且将 config/checkpoint SHA256、阈值、NMS/max-per-image 和输入 manifest 写入旁路 metadata；这属于导出既定候选预测，不能用结果修改模型或规则。

## 6. 稿件盘点

`/Users/mac/Desktop/lunwenxiezuo/EAAI_Paper/main.tex` 是完整的 Elsevier 单文件脚手架，不是空目录：已有背景/相关工作、OBB输出边界、数据与结果表结构、draft pipeline 图以及明确的“非闭环”限制。它没有可提交的结果：贡献条目、数据协议、对比/消融表、定量结果、图例、结论和声明仍为 placeholder/draft；当前稿件也尚未实现本报告的“分量可靠性—连续输出”实验叙事。`YSU_Thesis/document.tex` 是大论文资产，应保持深度、外参、物理摆角等后半部分与小论文分开。

## 7. 下一授权步骤

收到上述 V4+seq11-v2 source-side完整导出后，先验证 manifest/哈希/逐帧对齐与 OOF 或独立 calibration 的职责，再做只读错误支持和 score-only risk-coverage 基线。只有在相同 coverage 下出现可复现的分量风险降低，才进入可靠性方法实现；否则如实保留“检测输出可审计、可靠性新增贡献尚未成立”的结论。
