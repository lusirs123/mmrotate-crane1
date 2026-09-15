# SymEOOD–DINO 检测融合与 seq11 补充数据完整交接

更新日期：2026-09-05  
用途：供新对话窗口恢复本轮工作。本文记录本轮有效结果、失败路线、代码入口、测试产物、证据边界与当前唯一建议。

> 先读结论：目前没有一个新模型通过 fixed TEST，也没有未知序列或部署授权。最新 V4+seq11-v2 的 `epoch_10` 只是官方 738 帧 source-val 上的**暂定保留候选**。它在选取时考虑了 DFR，但 DFR 不是单项最优，也不是主排序项；现有 48 帧 seq11 辅助验证集没有覆盖困难帧，必须先补做时序块 OOF/CV，不能直接把 epoch10 写成正式最优模型。

## 1. 当前模型、数据和证据边界

### 1.1 研究对象

- 单目、单类、纯视觉 OBB 观测；检测框只包围抓斗**顶梁**，不包围抓斗平台。
- 本轮 seq11 标注采用 `k0=1.9`。
- 不使用 PLC、编码器或其他传感器。抓料阶段可以由作业流程中的人工接管解释为“不要求模型输出可用摇摆角”，但不能用人工逐帧挑选结果。
- 抓料/未完全离料阶段必须按统一操作规则标记为 `measurement-invalid`，不能因为某一帧恰好检测成功就保留。

### 1.2 必须分开的证据层级

1. `source-train`：只用于训练。
2. 官方 `source-val`：738 帧，用于模型选择与 source gate。
3. seq11 同视频辅助验证：只能验证同一视频内的机制支持，不能证明未知序列泛化。
4. fixed TEST：可用于固定模型的基准比较；如果依据同一 TEST 反复修改模型，它就不再是“唯一、无偏最终测试”。
5. 未知独立视频：当前没有，不能声称未知序列泛化或部署有效。

### 1.3 当前主要模型关系

```text
冻结 SymEOOD K1（epoch_24）FPN/检测头
                 +
预计算/冻结 DINO 候选
                 +
K1 anchored causal phase geometry refiner
                 ↓
同一 forward 处理 real 和 sim；无 domain/sequence/frame 路由
```

当前实验模型不是“real 用 DINO、sim 用 SymEOOD”的最终域特化模型。后续 refiner 路线要求 real/sim 共用同一 forward，DINO 缺失时统一回退 K1。

## 2. epoch10 是否考虑了 DFR

### 2.1 简短答案

考虑了，但要准确表述为：

- DFR 是**稳定性约束和候选间次级比较项**；
- epoch10 不是 real DFR 最优，也不是 sim DFR 最优；
- epoch10 被暂定保留，是因为它同时满足了当前最关键的多指标组合：两域 `MCML=0`，real/sim RIoU 均高于 Base V3，且 sim A-RMSE 略优于 Base V3；
- 当前打包产物没有包含可直接重算的 Base V3 DFR 对照，因此不能仅凭该压缩包宣称“epoch10 的 DFR 相对 Base V3 已正式非退化”。

### 2.2 官方 738 帧逐 epoch 指标

| epoch | real RIoU | real DFR↓ | real ACI↑ | real MCML↓ | sim RIoU | sim DFR↓ | sim ACI↑ | sim A-RMSE↓ | sim MCML↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7753 | 2.7960 | 0.9414 | 0 | 0.8884 | 2.2134 | 0.9551 | 7.0519 | 0 |
| 2 | **0.7895** | 2.7478 | 0.9409 | 0 | **0.8971** | 2.1785 | 0.9550 | 7.0549 | 0 |
| 3 | 0.7482 | 2.7885 | 0.9408 | 0 | 0.8475 | 2.1526 | 0.9553 | 9.8489 | 1 |
| 4 | 0.7853 | 2.7122 | 0.9418 | 0 | 0.8790 | 2.1809 | 0.9562 | 7.1324 | 0 |
| 5 | 0.7874 | 2.6130 | **0.9422** | 0 | 0.8708 | 2.1652 | 0.9564 | 7.0962 | 0 |
| 6 | 0.7682 | **2.5174** | 0.9412 | 0 | 0.8638 | 2.1626 | 0.9561 | 5.8379 | 0 |
| 7 | 0.7756 | 2.5925 | 0.9417 | 0 | 0.8818 | 2.1831 | 0.9563 | **5.7905** | 0 |
| 8 | 0.7760 | 2.6302 | 0.9416 | 0 | 0.8877 | 2.1703 | 0.9565 | 7.0225 | 0 |
| 9 | 0.7806 | 2.6664 | 0.9418 | 0 | 0.8917 | **2.1496** | **0.9566** | 7.0264 | 0 |
| 10 | 0.7828 | 2.6650 | 0.9418 | 0 | 0.8937 | 2.1639 | **0.9566** | 5.7957 | 0 |

所有 epoch 的 `real/R_center=100%`、`real/TDR_w10=100%`；除 epoch3 外，sim MCML 也为 0。

### 2.3 为什么暂定 epoch10，而不是 DFR 最优 epoch6/9

Base V3 epoch9 的已验证 source-val 对照为：

- real RIoU `0.7758`
- sim RIoU `0.8909`
- sim A-RMSE `5.8378°`
- real/sim MCML 均为 `0`

对比：

- epoch6：real DFR 最好，但 real RIoU `0.7682`、sim RIoU `0.8638`，几何损失明显。
- epoch9：sim DFR 最好，RIoU 也通过，但 A-RMSE `7.0264°` 明显变差。
- epoch2：两域 RIoU 最好，但 A-RMSE `7.0549°` 明显变差。
- epoch7：A-RMSE 最好，但 sim RIoU `0.8818` 低于 Base V3。
- epoch10：real RIoU `+0.0070`、sim RIoU `+0.0028`、A-RMSE `-0.0421°`，两域 MCML 保持 0；DFR 虽非最优，但没有出现前期双塔方案的 4–8 级别恶化。

因此 epoch10 是**多目标折中候选**，不是“DFR 最优 epoch”。正式表述应为：

> 在官方 738 帧 source-val 上，epoch10 是当前同时满足 Base V3 几何/角度非退化与 MCML 保持的暂定候选；DFR 被作为稳定性约束考虑，但未被单独最小化。

## 3. 基线指标与容易混淆的口径

### 3.1 K1 官方 source-val（738 帧）

| 域 | R_center | mean RIoU | DFR | ACI | A-RMSE | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 98.67% | 0.7851 | 3.4797 | 0.9396 | — | 1 |
| sim | 99.02% | 0.8891 | 2.0363 | 0.9565 | 17.8578° | 2 |

### 3.2 K1 fixed TEST（用户正式记录）

| 域 | R_center | mean RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 99.27% | 0.8046 | 2.5786 | 0.9393 | — | 80.85% | 44 |
| sim | 100% | 0.9087 | 2.0925 | 0.9576 | 1.6520° | 100% | 0 |

这两组数字来自不同 split，不能直接混合比较。之前出现的“K1 指标不一致”就是因为把 source-val 与 TEST 记录混在了一起。

## 4. 路线 A：固定 V4 输出级路由与抓料阶段审计

### 4.1 V4 定义

```text
real：DINO 有框则采用 DINO native OBB，否则回退 SymEOOD
sim：采用 SymEOOD
```

该 V4 是固定输出级组合，不是联合训练模型，也不是最终推荐结构。

### 4.2 fixed-target V4 结果

输入：992 帧 all-lane JSON；输出来源 `dino_native=420`、`sym_eood=572`。

| 域 | R_center | mean RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 95.95% | 0.6789 | 7.7873 | 0.8587 | — | 100% | 6 |
| sim | 100% | 0.8857 | 1.8932 | 0.9621 | 1.5487° | 100% | 0 |

- 相对 SymEOOD：`gained=162`、`lost=5`。
- 原 gate 只因 `real MCML=6 > 5` 失败。
- 两通道 oracle 在 real_seq03 的最长失败仍是 140–145 共 6 帧，说明这 6 帧并非再调两分支选择规则就能解决。
- 虽然用户允许工程门槛放宽为 6，但 V4 的 real RIoU、DFR、ACI 明显下降，因此不能仅凭 MCML 接近门槛把 V4 当最终模型。

### 4.3 抓料阶段 measurement-validity

操作规则把 real_seq03 `130–187`（共 58 帧）统一标为 `material_contact_grabbing_or_not_fully_detached`。这 58 帧中有 48 个 hit、10 个 miss；不能只删除失败帧。

排除后的 measurement-valid real 指标：

- frame：`362/420`，有效比例 `86.19%`
- R_center：`96.69%`
- mean RIoU：`0.6958`
- DFR：`7.5527`
- ACI：`0.8562`
- MCML_max：`4`

结论：measurement-valid gate 通过，但原始 gate 仍失败；且排除抓料阶段没有解决 V4 的几何与时序稳定性问题。论文可以并列报告 raw 与 measurement-valid，并说明抓料阶段由作业人员控制，但不能用该排除代替模型改进。

主要服务器产物：

```text
work_dirs/crane_symeood_dino_application_domain_v4/fixed_target_json_audit.json
work_dirs/crane_symeood_dino_conservative_takeover_v2/full_test_metric_v2/conservative_takeover_fusion_audit.json
```

## 5. 路线 B：source-val 几何可修正性审计

工具：

```text
crane_project/tools/symeood_dino_geometry_refinability_audit.py
tests/test_symeood_dino_geometry_refinability_audit.py
```

关键结果（738 帧 source-val）：

- DINO presence：`737/738=99.86%`；real 为 `226/226=100%`。
- real GT 中心在 DINO 框内：`100%`。
- real 中心误差 median `5.60 px`，p90 `12.58 px`。
- real both-present 223 帧：SymEOOD RIoU `0.7957`，DINO `0.6667`。
- DINO + Sym 尺寸：`0.7717`；DINO + Sym 角度：`0.6650`。
- DINO 中心 + GT 几何的 real oracle：`0.8578`。
- candidate oracle：real RIoU `0.7982`、DFR `4.0537`、ACI `0.9260`、MCML `0`；sim RIoU `0.8927`、DFR `2.1214`、ACI `0.9555`、MCML `0`。

有效结论：DINO 能提供存在性与中心支撑，主要几何问题是尺度，尤其宽度。该审计授权训练 source-only refiner，不授权 TEST、部署，也不支持继续训练 takeover router。

## 6. 路线 C：单帧 Size-only / Full geometry refiner

### 6.1 实现目的

- DINO 保持目标发现能力；
- 从冻结 SymEOOD FPN 对 DINO proposal 做 Rotated ROIAlign；
- 预测局部中心、log 尺寸和双角度残差；
- trainer 与 runtime 共用同一个 refiner、coder、active-component mask 和 decode；
- real/sim 同一 forward；DINO missing 统一回退 SymEOOD；不使用域路由。

主要代码：

```text
mmrotate/models/roi_heads/dino_conditioned_geometry_refiner.py
mmrotate/models/detectors/symeood_dino_geometry_refiner_trainer.py
mmrotate/datasets/pipelines/loading.py
mmrotate/core/hooks/geometry_refiner_contract_hook.py
crane_project/tools/symeood_dino_geometry_refiner_source_smoke.py
crane_project/tools/symeood_feature_reuse_equivalence.py
crane_project/configs/crane_symeood_dino_geometry_refiner_size_source_v1.py
crane_project/configs/crane_symeood_dino_geometry_refiner_full_source_v1.py
```

### 6.2 已锁定的技术合同

- 中心残差使用 DINO 局部归一化坐标。
- `le90 + edge_swap + proj_xy`，处理宽高交换等价框。
- 角度零初始化使用 `sin(2Δθ)` 与 `cos(2Δθ)-1`，零输出严格等于 DINO。
- DINO 原图框在增强前注册为 rotated bbox field，和 GT 同步 resize/flip。
- 冻结 baseline 参数与 BN buffer，训练前后哈希一致。
- `simple_test_from_features()` 复用一次 FPN，关闭 refiner 时必须与旧 `simple_test()` 逐元素等价。

### 6.3 结果与裁决

- Size-only：训练与冻结合同通过，但无法同时恢复 RIoU、DFR、ACI，未成为正式候选。
- Full：部分 R_center/ACI 改善，但 DFR 与 sim A-RMSE 明显不理想，未通过最终 source gate。
- 结论：几何容量审计成立，但一个单帧 ROI 残差头没有把 oracle 容量转化为可泛化性能。

本轮外部结果文件：

```text
20260828_092559.log.json
source_val_epoch10_results.pkl
source_val_epoch11_results.pkl
geometry_refiner_frozen_contract.json
```

## 7. 路线 D：Dual-tower V2 / V2.1

### 7.1 设计

- 把 Size-only 与 Full 的 size/pose 分支组合成双塔；
- V2.1 继续训练 size 分支，希望在保留中心/角度的同时改善尺度与 DFR。

代码与工具：

```text
crane_project/configs/crane_symeood_dino_geometry_refiner_dual_tower_source_val_v2.py
crane_project/configs/crane_symeood_dino_geometry_refiner_dual_tower_size_source_v21.py
crane_project/configs/crane_symeood_dino_geometry_refiner_dual_tower_v21_fixed_test.py
crane_project/tools/symeood_dino_dual_tower_v2_audit.py
crane_project/tools/symeood_dino_dual_tower_v2_package.py
crane_project/tools/symeood_dino_dual_tower_v21_source_gate.py
crane_project/tools/symeood_dino_dual_tower_v21_promote.py
```

### 7.2 V2.1 source-val 最好候选

epoch7 的 composite 最高 `0.0093115`：

- real RIoU `0.7521`、DFR `4.4628`、ACI `0.9253`
- sim RIoU `0.8052`、DFR `4.6079`、ACI `0.9075`

虽通过当时的宽松 composite gate，但改善量很小，且绝对几何/时序指标仍差。fixed TEST 进一步出现 `MCML=22` 和较高 DFR，因此关闭为最终路线。

本轮外部产物：

```text
dual_tower_v2_package.json
source_val_dual_tower_v2_results.pkl
source_val_dual_tower_v2_audit.json
dual_tower_v21_source_val_review.tar.gz
dual_tower_v21_fixed_test_results.pkl
```

## 8. 路线 E：因果历史支持审计与历史 refiner

### 8.1 支持审计

source-val：

- real 当前 miss 11 帧；1–4 帧历史 own-hit 支持率均为 100%，直接 hold 支持 `6/11=54.5%`。
- real 常速度支持 `5/11=45.5%`。
- sim 当前 miss 2 帧；2–4 帧历史可找到 own-hit，但简单 hold 为 0。

fixed-target：

- real 当前 miss 26 帧；历史窗口 1→4 时 own-hit 支持率从 `65.4%` 到 `88.5%`。
- 4 帧历史 oracle 可把 MCML `6→3`，但它是 GT oracle，不是部署结果。
- 常速度只支持 `5/26=19.2%`，不能降低 MCML。

审计文件：

```text
source_val_causal_history_support_audit.json
fixed_target_causal_history_diagnostic.json
crane_project/tools/symeood_dino_causal_history_support_audit.py
```

### 8.2 历史 refiner 实现

主要代码：

```text
crane_project/configs/crane_symeood_dino_causal_history_refiner_source_v1.py
crane_project/tools/symeood_dino_causal_history_source_smoke.py
crane_project/tools/symeood_dino_causal_history_source_gate.py
```

采用因果历史 ROI、当前帧 anchor、无 sequence/frame 身份输入、无域路由；无有效历史时严格退化为 current-only。正式 smoke 使用普通 K1：

```text
work_dirs/crane_symeood_k1/epoch_24.pth
```

而不是 BrightAug `epoch_20.pth`。smoke 最终通过，单卡 GTX1080 峰值 allocated 约 1.11 GiB、reserved 约 1.22 GiB。

## 9. 路线 F：K1-anchored causal V2 与 retentive V3

### 9.1 V2 fixed TEST

| 域 | R_center | RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 95.24% | 0.7001 | 5.9190 | 0.9051 | — | 97.46% | 19 |
| sim | 100% | 0.8908 | 1.9488 | 0.9603 | 1.5378° | 100% | 0 |

结论：sim 保持较好，但 real MCML=19、DFR 高、几何下降，失败。

主要产物：

```text
checkpoint_eval_summary.json
epoch10_source_promotion.json
fixed_test_audit.json
fixed_test_results.pkl
```

### 9.2 V3 source gate

V3 引入 K1 几何 anchor、bounded residual、continuous retention 和相邻帧误差一致性。10 个 epoch 中仅 epoch9 通过当时的完整 source gate：

- K1→V3：real RIoU `0.7690→0.7758`
- sim RIoU `0.8838→0.8909`
- sim A-RMSE `9.0415°→5.8378°`
- sim MCML `2→0`

该结果只允许 source promotion，不代表 fixed TEST、未知序列或部署。

### 9.3 V3 fixed TEST 与四组件消融

- epoch9 fixed TEST 没有复现 source-val 的改善，real DFR 仍差。
- full/current-only/center-only/K1-identity 四模式在 TEST 上只有约 `0.00x` 的浮动，不能称为有意义改进。
- anchor/fallback attribution 表明这是结构性问题，而不是某个历史分量的单独开关问题。

外部产物：

```text
v3_source_val_review.tar.gz
v3_epoch9_fixed_test_review.tar.gz
v3_component_fixed_test_review.tar.gz
source_val_anchor_fallback_attribution.json
fixed_target_anchor_fallback_attribution.json
```

结论：历史信息不是天然免疫域差。历史网络仍会学习 source 的运动、外观、相机抖动和失效模式；训练/TEST 视频差异会导致 history gate 和残差预测失配。

## 10. 路线 G：seq11 初始 59 帧 pilot

### 10.1 数据清理

磁盘曾显示 118 图像和 118 标注，但其中 59 个是 macOS AppleDouble `._*` sidecar。真实样本是 manifest 中的 59 帧，不是 118 帧。

初始 source inventory：

- frames：59
- SymEOOD hit rate：`0.4915`
- DINO hit rate：`0.9322`
- router support：不满足，证据用途为 `RETENTION_AND_NONREGRESSION_ONLY`

### 10.2 旧 48/11 block split

辅助 val 11 帧：

- K1 hit：4
- K1 missing + DINO hit：7
- K1 present-wrong + DINO hit：0

因此 `target_support=0`，裁决 `STOP_SEQ11_AUX_SUPPORT_INSUFFICIENT`。这不是路径错误，而是该 11 帧不支持“有框但框错”的接管监督。

### 10.3 59 帧 E1 fixed-target 诊断

将 59 帧加入训练后的诊断结果：

| 域 | R_center | RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 93.57% | 0.6763 | 5.8061 | 0.8951 | — | 98.98% | 13 |
| sim | 100% | 0.8753 | 2.0235 | 0.9579 | 1.6343° | 100% | 0 |

它说明新增困难数据有一定作用，但数量太少、real 几何与 DFR 仍差，不能作为最终结果；而且同视频数据不提供未知序列证据。

## 11. 路线 H：seq11-v2 扩充 251 帧与 V4 replay

### 11.1 数据事实

服务器目录：

```text
crane_project/data/crane_grab/extra_source_real_seq11_pilot_k1p9_v2
```

- 看到的 502 图像/JSON/TXT 文件 = 251 个真实样本 + 251 个 `._*` sidecar；sidecar 不是样本。
- 有效 251 帧均采用 `k0=1.9`、top-beam-only 标注。
- manifest 划分：train 203、aux-val 48、文件 overlap 0。
- 物化目录使用 hardlink；不把同一帧同时放进 train/val/test。

数据/审计产物：

```text
crane_project/data/crane_grab/extra_source_real_seq11_pilot_k1p9_v2/split_manifest.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/full_source_contract.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/blocksplit_v2/audited_split_materialization.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/blocksplit_v2/train_all_lane_audit.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/blocksplit_v2/aux_val_all_lane_audit.json
```

### 11.2 V4 replay 训练设计

- Base V3 epoch9 初始化并作为 teacher，retention loss weight `0.25`。
- 原始 source 2781 帧 replay + seq11 train 203 帧，总 `2984` 帧。
- 采样节奏 original:aux=`14:1`，防止单个视频主导训练。
- 固定每 epoch 1391 optimizer steps，共 10 epochs、13910 steps。
- seq11 使用相邻帧监督；不把 sequence/frame ID 输入模型。
- real/sim 共用同一 forward；无 domain routing。
- baseline/teacher 冻结，训练前后哈希一致。
- 单 GPU0；GTX1080 peak allocated `2,348,764,160 B`（约 2.19 GiB），reserved `2,512,388,096 B`（约 2.34 GiB）。

配置与 smoke：

```text
crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay.py
crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_aux_val.py
crane_project/configs/crane_symeood_k1_seq11_v2_aux_val_eval.py
crane_project/tools/symeood_dino_causal_history_source_smoke.py
```

训练目录：

```text
work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407
```

### 11.3 训练和 official source-val 结果

- smoke：`ALLOW_K1_RETENTIVE_SEQ11_V2_REPLAY_SOURCE_TRAINING`
- full coverage：原 2781 与 aux 203 在 10 epoch 中均被覆盖。
- 学习率：base `5e-5`；epoch1 warmup，2–6 约 `5e-5`，7–9 降至约 `1e-5`，epoch10 约 `5e-7`。日志显示为 `0.0` 是格式化，不是学习率真的为 0。
- 官方 738 帧结果见第 2 节；epoch10 为暂定多目标折中候选。

最新复核包：

```text
/Users/mac/Downloads/v4_seq11_v2_source_review.tar.gz
SHA256=d69484eb766d6e2f0f1fe59ec5f0c70623fd925d6f4a015ebe712dce61d85506
```

包内包含：10 个 epoch 在官方 738 帧和 seq11 aux-val 48 帧上的 `results.pkl`、`checkpoint_eval_summary.json`、训练日志、冻结合同、GPU 记录和 checkpoint SHA256；不包含大 checkpoint，因此压缩包较小是正常的。

### 11.4 当前关键缺陷：48 帧辅助验证不覆盖困难样本

251 帧全量类别：

- K1 hit：205
- K1 missing/wrong 且 DINO hit：32
- 两者都 bad：14

现有划分：

- train 203：K1 hit 157、hard 32、both-bad 14
- aux-val 48：K1 hit 48、hard 0、both-bad 0

因此：

- aux-val 48 上 K1 和 V4 各 epoch 都接近/达到 mAP 1.0，无法区分模型；
- 文件 overlap=0，所以不是直接帧泄漏；
- 但困难模式全部进入训练，形成**困难支持分配失败/选择盲区**；
- 不能用该 aux-val 为 epoch10 提供“seq11 困难机制有效”的证据。

## 12. 已修复的实现问题

以下属于工程修复，不是性能证据：

- `tools/test.py` config 缺少 `data.val`。
- `LoadDinoProposalFromAudit` 强制 all-lane 完整性。
- optimizer 把 `momentum` 传给 AdamW。
- `data.samples_per_gpu` 与 `data.train_dataloader.samples_per_gpu` 重复。
- frozen hash 在公开初始化前后取值时机不一致。
- `simple_test` 中 list/tensor 索引类型错误。
- `flip` 元数据缺失；增加显式 no-flip metadata。
- history `DataContainer.pad_dims` 触发 collate assertion。
- 误用 BrightAug `epoch_20`，改回普通 K1 `epoch_24`。
- dual-tower 推理时“至少一支可训练”的构造约束错误。
- child config 对 base `None` dict 合并时缺少 `_delete_=True`。
- seq11 `._*` AppleDouble 被误计为样本。
- split 名称的字符串前缀安全检查过严，改为规范化 data-root 子路径检查。
- Hook 的 `priority` 被写进 Hook 实例，触发 MMCV 保留属性错误。
- config 顶层保留打开的 JSON 文件句柄，导致 MMCV deepcopy `TextIOWrapper` 失败；改为局部读取并关闭。

这些修复只证明代码能够按合同运行，不能把 smoke/静态测试当成性能提升。

## 13. 本轮路线总裁决

| 路线 | 状态 | 可保留的结论 |
| --- | --- | --- |
| V4 real-DINO/sim-Sym 固定路由 | 失败为最终模型 | MCML 接近门槛，但 real 几何与时序明显下降 |
| 抓料阶段 measurement-validity | 有效的评估协议 | raw 与有效区间应双报告；不能代替模型优化 |
| takeover/router | 关闭 | source 正样本支持不足，不能安全学习接管 |
| geometry capacity audit | 有效 | DINO 存在性/中心可靠，主要误差为尺寸 |
| Size-only / Full refiner | 失败 | 可训练合同成立，但未把 oracle 容量转化为性能 |
| dual-tower V2/V2.1 | 失败 | source 宽松 gate 的小增益未在 TEST 成立，MCML=22 |
| 简单 hold / velocity | 仅诊断 | 历史有一定 support，但简单外推覆盖不足 |
| causal V2 | 失败 | sim 好、real MCML19/DFR高 |
| retentive V3 | source 有效、TEST 不稳 | epoch9 是 source promotion-only，域差仍明显 |
| seq11 初始 59 帧 E1 | 弱正向诊断 | 数据有用迹象，但量少且 MCML13 |
| seq11-v2 251 帧 V4 replay | 当前最值得继续 | official 738 上 epoch10 多指标小幅正向，但 aux-val 困难支持为 0 |

## 14. 当前唯一推荐的下一步

### 14.1 先保留，不再修改 epoch10

保留以下 checkpoint 作为**暂定 source-retention candidate**：

```text
work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407/epoch_10.pth
```

在完成下面的 OOF/CV 前：

- 不称其为正式最优；
- 不生成 `source_gate_passed=True` 的 promotion artifact；
- 不根据 fixed TEST 调 V4 参数；
- 不宣称未知序列泛化或部署。

### 14.2 对 251 帧做三折时序块 OOF/CV

目的不是增加随机拆帧，而是让全部困难帧至少一次处于未见验证块：

- 以连续时序块/困难事件为基本单元，不随机拆相邻帧；
- 三折中每折用约 2/3 seq11 块训练，剩余块验证；
- 32 个 `K1 missing/wrong + DINO hit` 和 14 个 `both-bad` 必须在 OOF 汇总中各出现一次；
- 每折仍 replay 原始 2781 帧并保留 Base V3 teacher；
- 官方 738 帧只做 non-regression gate；
- OOF 结果要报告困难帧 rescue、old-correct lost、RIoU、DFR、ACI、MCML，而不是只看 mAP。

仓库已有可复用的 block-CV 工具，不应再新造一套：

```text
crane_project/data_contracts/real_seq11_pilot_k1p9_three_window_block_cv_v1.json
crane_project/tools/symeood_dino_seq11_block_cv_materialize.py
crane_project/tools/symeood_dino_seq11_block_cv_gate.py
crane_project/tools/symeood_dino_seq11_block_cv_select.py
crane_project/tools/symeood_dino_seq11_block_cv_support_audit.py
crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v3_seq11_blockcv.py
```

这些旧工具需要先扩展到 251 帧 v2 manifest 与 V4 replay 合同，不能原样把旧 59 帧 contract 当新数据使用。

### 14.3 OOF 通过后的顺序

1. 生成 V4 专用 dual-source gate，明确 official-738 retention 与 seq11 OOF hard-support 两个部分。
2. 固定 epoch/模型，不再读取 TEST 做选择。
3. 生成带 checkpoint SHA256、数据合同和 `source_gate_passed=True` 的 promotion artifact。
4. 再运行 fixed TEST 基准，并同时报告：
   - raw 全帧指标；
   - 按预先定义的抓料阶段排除后的 measurement-valid 指标；
   - 抓料阶段被排除的总帧数、hit/miss 数和理由。
5. fixed TEST 只用于报告这个已固定模型，不能反向扫描阈值或继续选 epoch。

若三折 OOF 仍不能在困难块上稳定 rescue，当前结论应收束为：seq11 数据揭示了结构性 failure，但同视频补充数据不足以形成可泛化融合模型；需要新的独立真实视频，而不是继续在同一 TEST 上堆规则。

## 15. 新对话窗口最小读取清单

新对话按以下顺序读取：

1. 本文：`docs/20260905_symeood_dino_fusion_seq11_full_handoff.md`
2. 背景：`.hermes/desktop-attachments/背景.md`
3. 长期实验总账：`docs/innovation4_dino_small_target_progress.md`
4. 最新复核包：`/Users/mac/Downloads/v4_seq11_v2_source_review.tar.gz`
5. 最新 V4 replay config 与 frozen contract。
6. seq11-v2 full/train/val all-lane audit 与 split manifest。

恢复时必须先回答四个问题：

- 当前讨论的是 source-val、seq11 同视频 OOF，还是 fixed TEST？
- checkpoint 是否已经 source-gated，还是仅为训练/暂定候选？
- 指标中的 K1 是 source-val 还是 TEST 口径？
- 是否把抓料阶段 raw 指标和 measurement-valid 指标分开报告？

## 16. 资源约定

- 后续单卡命令优先使用物理 GPU0：`CUDA_VISIBLE_DEVICES=0 ... --gpu 0`。
- 多卡训练使用前三张卡：`CUDA_VISIBLE_DEVICES=0,1,2`。
- 每次训练或真实栈 smoke 记录 peak allocated、peak reserved、单帧/step 延迟和实际 forward count。
- 不预先声称 refiner 不增加显存；FPN 特征生命周期延长会增加峰值显存。

## 17. 本轮外部结果文件索引

下面按阶段列出用户在本轮对话中提供过的结果文件。部分文件只存在于当时的
`/Users/mac/Downloads` 或 Codex attachment 临时目录；迁移到新机器/新账号时需另行复制。

### 17.1 V4、source inventory 与 measurement-validity

```text
source_inventory_support.json
source_inventory_all_lane_audit.json
fixed_target_causal_history_diagnostic.json
```

服务器权威输入仍是：

```text
work_dirs/crane_symeood_dino_application_domain_v4/fixed_target_json_audit.json
work_dirs/crane_symeood_dino_conservative_takeover_v2/full_test_metric_v2/conservative_takeover_fusion_audit.json
```

### 17.2 单帧 geometry refiner

```text
20260828_092559.log.json
20260828_135345.log.json
source_val_epoch10_results.pkl
source_val_epoch11_results.pkl
geometry_refiner_frozen_contract.json
cuda_peak_memory_rank0.json
```

### 17.3 Dual-tower

```text
dual_tower_v2_package.json
source_val_dual_tower_v2_results.pkl
source_val_dual_tower_v2_audit.json
dual_tower_v21_source_val_review.tar.gz
dual_tower_v21_fixed_test_results.pkl
```

### 17.4 Causal V2/V3

```text
source_val_causal_history_support_audit.json
20260829_142925.log.json
v2_source_val_review.tar.gz
checkpoint_eval_summary.json
epoch10_source_promotion.json
fixed_test_audit.json
fixed_test_results.pkl
v3_source_val_review.tar.gz
v3_epoch9_fixed_test_review.tar.gz
v3_component_fixed_test_review.tar.gz
source_val_anchor_fallback_attribution.json
fixed_target_anchor_fallback_attribution.json
```

其中 `checkpoint_eval_summary.json` 和 `fixed_test_results.pkl` 在不同 work dir 中会重名；必须和
所属 config、checkpoint SHA256、工作目录一起读取，不能只凭文件名归因。

### 17.5 seq11 E1 与 seq11-v2

```text
v3_vs_seq11_e1_epoch10_fixed_target_pair.tar.gz
v4_seq11_v2_source_review.tar.gz
```

最新包的绝对路径和 SHA256：

```text
/Users/mac/Downloads/v4_seq11_v2_source_review.tar.gz
d69484eb766d6e2f0f1fe59ec5f0c70623fd925d6f4a015ebe712dce61d85506
```
