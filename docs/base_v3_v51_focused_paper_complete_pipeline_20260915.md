# Base V3 + V5.1 小论文模型与实验完整流程

更新日期：2026-09-16
本地主仓库：`/Users/mac/Documents/paper/symEOOD`  
服务器主仓库：`/media/omnisky/personal_files/ljj/symEOOD`  
服务器环境：Python 3.8，conda 环境 `mmrotljj`

## 1. 文档用途

本文档固化当前小论文范围内已经实现并实际运行的模型链、评估链、结果文件、统计口径、主要结论、证据限制和复现方法。它是模型与实验交接文档，不是论文正文。

当前研究问题是：在复杂港口图像序列中，如何检测抓斗，并判断检测框的中心、尺度和方向是否足以作为可靠的连续视觉观测。

已实现的整体路线为：

```text
港口 RGB 图像序列
  → 冻结 DINOv2 native-S14 候选
  → SymEOOD K1 epoch-24 有向检测
  → Base V3 epoch-9 因果细化
  → V5.1 中心/尺度/方向分量可靠性判定
  → 观测值 + 有效性状态连续输出
  → 推理结束后连接 GT
  → 完整框、分量可靠性和时序指标报告
```

这条链已经在固定 TEST 的 992 帧上完成一次服务器完整运行。这里的“完整”指工程入口、在线输出、后验评估和报告生成均已连通；它不表示论文所需的全部对比实验、消融实验和独立泛化实验已经齐备。

## 2. 研究范围与证据边界

### 2.1 当前纳入范围

- 单类抓斗有向目标检测。
- OBB 中心、长短边尺度、方向和检测分数的输出。
- 中心、尺度、方向三个分量的独立有效性判定。
- `measurement`、`prediction`、`unavailable` 三种观测状态。
- 完整 OBB 与部分分量观测的区分。
- 固定 TEST 上的完整框、分量误差、缺测连续性和处理耗时统计。
- raw、仅置信度拒绝、V5.1 三种输出策略的并列比较。

### 2.2 当前不纳入范围

- 不使用 PLC。
- 不把图像平面 OBB 直接解释为三维位置、物理摆角或绳长。
- 不进行 EKF 状态估计、防摇控制或风险预警闭环。
- 不把有限历史值保持解释成运动预测。
- 不根据已经暴露的固定 TEST 调整阈值、排序特征、在线策略或 checkpoint。
- 不根据这批结果宣称未知序列泛化、校准不确定性或部署实时性。

### 2.3 数据证据等级

本次结果来自已经暴露的固定 TEST，报告协议明确记录：

- `fixed_test_previously_exposed = true`
- `parameter_tuning_authorized = false`
- `unknown_sequence_claim_authorized = false`
- `physical_state_claim_authorized = false`

因此，本批结果可以描述冻结实现对这组固定输入的表现，可以用于结果分析和失败模式讨论，不能再用于选择参数并随后把同一 TEST 当作独立验证集。

## 3. 统一入口与核心文件

统一配置入口：

```text
crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json
```

统一执行入口：

```text
crane_project/tools/base_v3_obb_focused_paper_pipeline_v1.py
```

固定 TEST 自定义指标实现与 V1 历史审计入口：

```text
crane_project/tools/analyze_unified_full_run_v1.py
```

自定义指标已经接入统一入口生成的模型总报告；该脚本仍负责 V1 下载文件的固定 SHA256 审计和本地 GT 差异记录。

对应回归测试：

```text
tests/test_base_v3_obb_focused_paper_pipeline_v1.py
tests/test_analyze_unified_full_run_v1.py
tests/test_eval_crane_offline_records.py
```

当前本地源码身份如下。代码继续修改后必须重新计算，不能把这些 SHA256 当作永久常量：

| 文件 | 当前本地 SHA256 |
| --- | --- |
| `base_v3_obb_focused_paper_pipeline_v1.json` | `373228fd49f82cd029c45830a6a274e87082253bfd465824d2600d6871594082` |
| `base_v3_obb_focused_paper_pipeline_v1.py` | `f8974eb33e9e4c98bff0550ec8a0ed17654a47702d168a5e5de17b2979ae8253` |
| `analyze_unified_full_run_v1.py` | `fe63b0f7269a74a83537bf299e44c5a899d63c3fdb870212d9d0e8b108d22eb2` |
| `base_v3_obb_true_online_finalization_v2.py` | `9b27215f614ae473f54256c834232221dcd7d38c4616c6af840d6c89e9c6b935` |
| `test_base_v3_obb_focused_paper_pipeline_v1.py` | `38af8f5b2e3d745a93a5d690c643d77b1494e99d2249c2e3c649fc7fa5301ffa` |
| `test_analyze_unified_full_run_v1.py` | `71f15d5c758c1901be025b330078e2e75a38a188dc01a4be835aeebf967df2b9` |

## 4. 模型链逐级说明

### 4.1 输入和时间顺序

输入为：

```text
crane_project/data/crane_grab/test/images
```

本次运行共 992 帧：

| 数据域/序列 | 帧数 |
| --- | ---: |
| real | 420 |
| sim | 572 |
| seq02 | 220 |
| seq03 | 200 |
| seq09 | 572 |

模型按序列和帧号顺序处理图像。历史窗口长度为 4，只保存严格早于当前帧的 DINO top-1 候选；序列变化或帧号不连续时清空历史。在线决策不读取标注或 GT，不读取未来帧，域身份不参与模型决策；序列和帧号只用于排序与历史重置。

数据没有可绑定时间戳，因此所有连续缺失长度以“帧”报告，不根据假定帧率换算为秒。

### 4.2 冻结 DINOv2 候选

配置：

```text
crane_project/configs/crane_symeood_formal_dino_native_s14_v1.py
```

主要模型身份：

| 资产 | SHA256 |
| --- | --- |
| DINOv2 ViT-L/14 backbone | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |
| source-safe interpolated head | `6a4e112f89f92744dbb8f7a0725af185afd110b3aa18414d28340f31d7854c2a` |
| DINO 配置 | `00175d07197679d09414d614985a041fe6d59f59bcd8e0e0ebbffc2b8db069d5` |

DINO 候选既用于当前候选融合，也向严格因果历史提供 top-1 proposal。它在当前 992 帧中均产生候选。

### 4.3 SymEOOD K1 检测器

配置：

```text
crane_project/configs/crane_symeood_k1.py
```

checkpoint：`work_dirs/crane_symeood_k1/epoch_24.pth`  
checkpoint SHA256：`57233e5423de4a9d0a67fd51058cd7d92adaac5d6621f275cc0ec5b0fc7f9ee2`

K1 在 992 帧中的 846 帧提供主锚框；另外 146 帧使用 DINO fallback。报告中的 `anchor:k1` 与 `anchor:dino_fallback` 分组就是按该来源统计。

### 4.4 Base V3 epoch-9 因果细化

在线配置：

```text
crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_online_v1.py
```

细化器 checkpoint：

```text
work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v3_seed3407/
k1_retentive_v3_epoch9_promoted.pth
```

checkpoint SHA256：`28d17d91fe5a9ae2f548d9b8f7fc9c4fbd8a53382ed7f12b6f0446a6064dce85`

Base V3 在全部 992 帧输出 OBB。报告中的 `raw` 是 Base V3 前端 OBB 输出，不是原始图像、未处理预测或基础检测器。Base V3 输出中的数值 `1.0` 具有 `constant_refiner_output_not_confidence` 语义，不能作为常规置信度解释。置信度拒绝使用的是锚框来源分数。

### 4.5 V5.1 分量可靠性输出

V5.1 冻结校准文件：

```text
work_dirs/base_v3_obb_reliability_baseline_v1/
base_v3_obb_hybrid_observation_calibration_v51.json
```

SHA256：`626a8cf4727181fa91ee222c1654a50fa9fffafaff9dca20e693df25e9e94d7e`

三个分量分别处理：

- 中心：按冻结锚框置信度门限接收或拒绝。
- 尺度：按冻结 V5.1 监督尺度风险接收或拒绝。
- 方向：按冻结锚框置信度接收；当前测量不可用时允许最多保持最近一次有效方向 1 帧。

每个分量输出：

- `value`：中心坐标、长短边或方向值；不可用时为 `null`。
- `valid`：在线规则是否允许使用该分量。
- `state`：`measurement`、`prediction` 或 `unavailable`。
- `source`、`risk`、`gate`、`reason`、距最近测量的帧龄等审计信息。

其中 `prediction` 只表示有限的最近测量值保持，不表示模型完成了速度估计、轨迹预测或物理状态估计。

## 5. 三种比较方法

| 方法 | 定义 | 用途 |
| --- | --- | --- |
| `raw` | Base V3 的中心、尺度、方向全部作为测量输出 | 完整前端性能基线 |
| `score_rejection_only` | 三个分量统一按锚框置信度接收或拒绝 | 单纯置信度筛选基线 |
| `v51_hybrid` | 中心按分数、尺度按冻结风险、方向按分数并允许 1 帧保持 | 分量可靠性方法 |

V5.1 的目标是允许分量级输出，而不是保证每帧都有完整 OBB。只要中心、尺度、方向中有一项无效，该帧就不能计为完整 OBB；其余有效分量仍可以单独输出。

## 6. 在线推理与后验评估隔离

统一入口现在分为三个可独立执行的阶段：

```text
infer（无 GT） → evaluate（推理后连接 GT） → report（只读结果生成报告）
```

`full` 模式依次调用这三个阶段。执行顺序如下：

1. 读取 RGB 图像并进行严格时间顺序推理。
2. 生成在线 JSON；这一阶段不读取 GT。
3. 推理结束后加载固定标注及历史结果身份。
4. 将在线观测与 GT 对齐，计算逐帧中心、尺度、方向误差和完整 OBB RIoU。
5. 导出或核对长格式观测 CSV。
6. 为本次在线 JSON 生成 finalization 运行时契约，并绑定其实际 SHA256。
7. 生成 finalization JSON 后，为实际 finalization、在线 JSON 和 CSV 生成论文指标运行时契约。
8. 生成论文指标 JSON。
9. 保存诊断输入及诊断运行时契约，再从本次逐帧记录生成 V3.1-r1 可靠性诊断。
10. 从同一绑定结果自动计算 R_center、A-RMSE、DFR、ACI、TDR、MCML、MRF。
11. 生成精简模型总报告、精简可靠性主报告、可靠性详细诊断和完整 JSON。

该隔离用于保证 GT 只参与后验度量，不进入在线判定。固定 TEST 已暴露这一事实仍然存在，所以隔离并不会重新赋予该 TEST 独立验证资格。

## 7. 输出状态语义

| 状态 | 含义 | 能否构成当前分量观测 |
| --- | --- | --- |
| `measurement` | 当前帧检测测量通过冻结规则 | 能 |
| `prediction` | 使用有限历史保持值；当前只出现在方向分量 | 能，但必须标记来源和帧龄 |
| `unavailable` | 当前规则拒绝且没有允许的保持值 | 不能 |

`valid=true` 仅表示冻结在线规则允许输出，不表示该值经 GT 证明正确。GT 正确性只在推理结束后的诊断中计算。

## 8. 指标体系与分母

### 8.1 完整 OBB 指标

- `available_obb_coverage`：中心、尺度、方向同时有效的帧数 / 全部帧数。
- `mean_available_riou`：仅在具有完整 OBB 的帧上计算平均 RIoU。
- `riou_hit_coverage`：具有完整 OBB 且 RIoU ≥ 0.5 的帧数 / 全部帧数。
- `riou_hit_rate_given_available`：RIoU ≥ 0.5 的完整框数 / 完整框数。
- `jointly_correct_coverage`：三个有效分量同时满足各自正确阈值的帧数 / 全部帧数。
- `false_all_components_valid_rate`：完整观测帧中至少一个分量不满足正确阈值的比例。
- `longest_incomplete_obb_run`：连续不能组成完整 OBB 的最大帧数。

冻结正确性阈值：中心误差 ≤ 5 px、尺度最大相对误差 ≤ 0.1、方向误差 ≤ 3°、RIoU 命中阈值为 0.5。

完整 OBB 覆盖率、联合分量正确率、RIoU 命中覆盖率和部分分量可用性是不同指标，不能相互替代。

### 8.2 分量指标

- `measurement_coverage`：当前测量状态帧数 / 全部帧数。
- `output_coverage`：测量状态和有限保持状态帧数 / 全部帧数。
- `mean_available_output_error`：仅在有效分量上计算的平均误差。
- `bad_available_output_rate`：有效分量中误差超过冻结正确阈值的比例。
- `correct_output_coverage`：有效且误差满足阈值的帧数 / 全部帧数。
- `longest_unavailable_run`：该分量连续不可用的最大帧数。
- `partial_valid_count`：仅一个或两个分量有效的帧数；这些帧不构成完整 OBB。

中心误差单位为像素。尺度误差为长边相对误差和短边相对误差中的较大值。角度误差使用 180° 周期对称的最小差，单位为度。

### 8.3 同覆盖率尺度比较

比较容差固定为：

```text
comparison_tolerance = 1e-12
```

统计修订标识为：

```text
v31_r1_absolute_tolerance
```

规则如下：

- 两个差值都 `< -1e-12` 才计为风险排序对分数排序的双指标严格胜出。
- 两个差值的绝对值都 `<= 1e-12` 才计为双指标平局。
- 一项改善、一项平局不计为双指标严格胜出。
- 原始浮点差值保留，不为显示平局而改写数值。

两个指标分别是保留样本平均尺度误差和尺度失败率。报告必须写“严格双指标胜出”，不能改写成“不劣且至少一项改善”。

### 8.4 自定义时序与检测指标

- `R_center(%)`：所有正 GT 帧上，完整 OBB 中心误差 `<15 px` 的比例；缺少完整 OBB 记失败。
- `mean_RIoU`：所有正 GT 帧上的平均 RIoU；缺少完整 OBB 记 0。它与 `mean_available_riou` 的分母不同。
- `A-RMSE(deg)`：仅在仿真域定义。完整框中心误差 `<10 px` 时使用周期角度误差；缺框或中心误差不满足条件时记 90° 惩罚。
- `DFR(%/frame)`：连续两个有效完整框对角线长度的相对变化。
- `ACI`：连续两个有效完整框方向变化相对 35° 归一化后的一致性。
- `TDR_w10(%)`：连续 10 帧窗口中至少有一次 RIoU ≥ 0.5 的窗口比例。
- `MCML_max(frames)`：连续 RIoU < 0.5 或无完整框的最长帧数。
- `MCML_mean(frames)`：每个连续物理片段的最大连续失配长度的平均值。
- `MCML_pass(limit=5)`：最大连续失配是否不超过 5 帧。
- `MRF(frames)`：发生失配后最终恢复的失配段平均长度；序列末尾未恢复段不纳入。

DFR 和 ACI 也包含物体真实运动，且筛选会改变参与计算的相邻帧对，因此不能单独作为几何准确性提升证据。A-RMSE 包含 90° 缺测惩罚，也不能当作有效方向输出的平均角度误差。

`DEP` 需要物理深度参照。当前纯视觉实验没有对应真值，不计算且不补造。

## 9. 本次服务器完整运行产物

目录：

```text
work_dirs/base_v3_obb_reliability_baseline_v1/unified_full_run_v1
```

原始服务器产物：

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| `unified_true_online_pipeline_v1.json` | 不读 GT 的逐帧在线观测 | `ef5487b9b20cc8f7723fd06f5004b330060f101637b3f68ebdbec23ac1a158ac` |
| `unified_true_online_finalization_v2.json` | 在线输出连接 GT 后的逐帧误差和 RIoU | `1d848597ac1488a4bb70d290b69bdb8ea1405cb8e7f1134f505fe4b9fa0ae88b` |
| `unified_true_online_observations_v2.csv` | 992×3×3 的长格式观测表，共 8928 行 | `8a9665c220a1138520a4bf34008f016b15f2b549d734d9819acbc5f8930704d3` |
| `unified_true_online_paper_metrics_v3.json` | 完整框、分量和来源分组结果 | `9d4860d82fc8d92cc75425da98ec2acd72294b33ecf8c4930890d5a46130d` |
| `unified_reliability_diagnostic_v31_r1.json` | 本次运行独立重建的 V3.1-r1 诊断 | `093de7db915308eb41170b3fcdc3f8f5f61d83f51dcca1df0e668988116ddb3d` |
| `fixed_test_focused_paper_model_pipeline_v1.json` | 机器可读模型总报告 | `c90011407e83d14704dd81b27e7cb390bdb66b609df89129c4850f488f6b3ca2` |
| `fixed_test_focused_paper_model_pipeline_v1.md` | 人工阅读模型总报告 | `436bd6755bcf82e1520ca49b393243d875126218004f6f8caf7ebcda2f439f16` |
| `fixed_test_focused_paper_reliability_v1.json` | 机器可读可靠性专项报告 | `3886713a021c7041b76cf38b969cb54b7b8ee3417403e76778c7213ca3d9614b` |
| `fixed_test_focused_paper_reliability_v1.md` | 人工阅读可靠性专项报告 | `5122959000cea0ebbf6560887fc1f0382b2aeb6b7baa7505a17cbc22d2a5e560` |

本地补充产物：

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| `verified_custom_metrics_v1.json` | 自定义指标、输入绑定和差异审计 | `ac2890a1d857bc49f87ec2d1bd86968d7eac395524a4e4776f593eb770520616` |
| `verified_custom_metrics_v1.md` | 包含自定义指标、模型总表和可靠性表的完整阅读报告 | `aa62c4a848cd993f2d301e1eba2626512d5d749a809d6df62e00345f0d5afa06` |

原始 9 份服务器文件均保持不变。补充文件使用新文件名，没有覆盖历史报告。

新分阶段入口在新运行目录中还会生成：

| 文件 | 用途 |
| --- | --- |
| `unified_inference_receipt_v1.json` | 绑定无 GT 在线 JSON、观测 CSV、命令和推理计时 |
| `runtime_true_online_finalization_contract_v2.json` | 绑定本次在线 JSON 的实际 SHA256 |
| `runtime_true_online_paper_metrics_contract_v3.json` | 绑定本次 finalization、在线 JSON 和 CSV 的实际 SHA256 |
| `runtime_reliability_diagnostic_input_v51.json` | 保存本次诊断使用的逐帧输入 |
| `runtime_reliability_diagnostic_contract_v51.json` | 绑定诊断输入和论文指标身份 |

这些文件解决了原 `full` 实现只在内存中改写 CSV 绑定、却仍在结果里引用模板契约身份的问题。模板契约保持冻结；每次新运行另存自己的运行时契约。

## 10. 核心结果

### 10.1 全体 992 帧的完整 OBB

| 方法 | 完整 OBB 覆盖率 | 有效完整框平均 RIoU | RIoU 命中覆盖率 | 联合正确覆盖率 | 完整观测中至少一分量错误比例 | 最长不完整段 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 1.000000 | 0.796358 | 0.947581 | 0.576613 | 0.423387 | 0 |
| score rejection only | 0.832661 | 0.827773 | 0.819556 | 0.542339 | 0.348668 | 19 |
| V5.1 | 0.793347 | 0.832538 | 0.780242 | 0.534274 | 0.326557 | 19 |

筛选提高了保留完整框的平均 RIoU，同时降低了完整框覆盖率、命中覆盖率和联合正确覆盖率。V5.1 的有效完整框平均 RIoU 最高，但其完整 OBB 覆盖率和联合正确覆盖率均低于仅置信度拒绝，所以不能称为整体最优。

### 10.2 all 分组的分量可靠性

| 方法 | 分量 | 输出覆盖率 | 有效输出平均误差 | 有效输出错误率 | 正确输出覆盖率 | 最长不可用段 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | center | 1.000000 | 3.17704 px | 0.219758 | 0.780242 | 0 |
| raw | scale | 1.000000 | 0.137320 | 0.351815 | 0.648185 | 0 |
| raw | angle | 1.000000 | 2.12265° | 0.187500 | 0.812500 | 0 |
| score rejection only | center | 0.832661 | 2.55079 px | 0.148910 | 0.708669 | 19 |
| score rejection only | scale | 0.832661 | 0.106951 | 0.279661 | 0.599798 | 19 |
| score rejection only | angle | 0.832661 | 1.87457° | 0.147700 | 0.709677 | 19 |
| V5.1 | center | 0.832661 | 2.55079 px | 0.148910 | 0.708669 | 19 |
| V5.1 | scale | 0.953629 | 0.138241 | 0.342495 | 0.627016 | 17 |
| V5.1 | angle | 0.912298 | 2.00040° | 0.167956 | 0.759073 | 18 |

V5.1 相对置信度筛选增加了尺度和方向覆盖率，也增加了有效输出中的错误比例和平均误差。它形成的是“更多部分分量可用”与“新增观测质量下降”之间的权衡。

V5.1 在 992 帧中产生：

- 尺度测量 946 帧，不可用 46 帧。
- 方向测量 826 帧、保持输出 79 帧、不可用 87 帧。
- 仅部分分量有效 202 帧，其中真实域 165 帧、仿真域 37 帧。
- 完整 OBB 覆盖率相对仅置信度拒绝下降 0.039315。
- 联合正确覆盖率相对仅置信度拒绝下降 0.008065。

### 10.3 角度保持

| 分组 | 保持帧数 | 改善 | 退化 | 平局 | 相对 raw 的平均角度误差变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 79 | 29 | 50 | 0 | +0.678074° |
| real | 48 | 15 | 33 | 0 | +1.063060° |
| sim | 31 | 14 | 17 | 0 | +0.081969° |
| seq02 | 24 | 11 | 13 | 0 | -0.679623° |
| seq03 | 24 | 4 | 20 | 0 | +2.805740° |
| DINO fallback | 7 | 6 | 1 | 0 | -4.752250° |
| K1 | 72 | 23 | 49 | 0 | +1.206020° |

局部分组存在收益，但 all、real 和 K1 主体分组整体退化。不能用 fallback 的 7 个保持样本代替全体结论。

### 10.4 同覆盖率尺度排序

all、real、sim 三个总体分组中，V5.1 尺度风险排序在 0.95、0.90、0.80、0.70、0.50 覆盖率下均没有对锚框分数排序形成双指标严格胜出。100% 覆盖率按 `1e-12` 绝对容差判为平局。

DINO fallback 子组在 0.80 和 0.70 覆盖率下出现局部双指标胜出，但该子组只有 146 帧，而且完整联合正确覆盖率为 0。这个局部诊断不能推广为尺度排序整体优于置信度排序。

### 10.5 自定义指标

真实序列完整框口径：

| 指标 | raw | score rejection only | V5.1 |
| --- | ---: | ---: | ---: |
| R_center(%) | 97.38 | 68.33 | 59.05 |
| mean_RIoU，缺框记 0 | 0.6831 | 0.5055 | 0.4375 |
| DFR(%/frame) | 6.1624 | 4.8030 | 5.4445 |
| ACI | 0.8909 | 0.9071 | 0.8926 |
| TDR_w10(%) | 96.69 | 93.38 | 90.84 |
| MCML_max(frames) | 22 | 33 | 33 |
| MCML_mean(frames) | 8.00 | 17.67 | 19.67 |
| MRF(frames) | 2.74 | 2.90 | 3.48 |

仿真序列完整框口径：

| 指标 | raw | score rejection only | V5.1 |
| --- | ---: | ---: | ---: |
| A-RMSE(deg) | 1.6190 | 22.9423 | 22.9423 |
| R_center(%) | 100.00 | 93.53 | 93.53 |
| mean_RIoU，缺框记 0 | 0.8795 | 0.8242 | 0.8242 |
| DFR(%/frame) | 2.0227 | 1.9744 | 1.9744 |
| ACI | 0.9589 | 0.9600 | 0.9600 |
| TDR_w10(%) | 100.00 | 100.00 | 100.00 |
| MCML_max(frames) | 0 | 4 | 4 |
| MCML_mean(frames) | 0.00 | 4.00 | 4.00 |

这里 V5.1 与仅置信度拒绝的完整框自定义指标在仿真域相同，是因为二者最终形成完整 OBB 的帧集合和框相同；V5.1 额外提供的部分分量不会进入完整框指标。

### 10.6 K1 与 DINO fallback 来源

K1 来源 846 帧，DINO fallback 来源 146 帧。raw 的来源分组结果：

| 来源 | 平均有效 RIoU | RIoU 命中覆盖率 | 联合正确覆盖率 |
| --- | ---: | ---: | ---: |
| K1 | 0.824793 | 0.957447 | 0.676123 |
| DINO fallback | 0.631592 | 0.890411 | 0.000000 |

fallback 的联合正确覆盖率为 0，含义是在这 146 帧中没有完整框的三个分量同时满足 5 px、0.1 和 3° 三项阈值。它不表示每个 fallback 框完全错误，也不表示每个单独分量都不正确。

### 10.7 运行时间

- 总时间：1905.976985 s。
- 平均：1.921348 s/frame。
- 平均吞吐：0.520468 frame/s。
- 未使用多 GPU 并行。
- 历史报告记录的范围标签：`full_command_model_init_detector_image_io_and_reporting`。

复核代码后确认，这个历史计时实际包围在线推理子进程，包含模型初始化、图像 I/O、检测与在线 JSON 生成，没有包含随后执行的 GT 后验评估和最终报告写入。旧标签的 `and_reporting` 过宽。新入口使用 `model_init_detector_image_io_and_online_json_generation`，并把计时写入 inference receipt。该速度仍不能当作稳态纯检测 FPS，也不能代表可靠性模块单独开销。

## 11. 复现与一致性审计

### 11.1 已确认事项

- 下载的 9 份服务器产物 SHA256 全部与服务器输出一致。
- 文件内部发现的 17 处本批产物绑定与预期 SHA256 一致。
- 观测 CSV 共 8928 行，等于 992 帧 × 3 方法 × 3 分量。
- 本次 `unified_true_online_pipeline_v1.json` 和 `unified_true_online_observations_v2.csv` 与此前保存的固定在线文件逐字节一致。
- 这证明在线观测流被再次生成一致，但也说明它仍是同一固定 TEST 证据，不是新数据验证。

### 11.2 当前运行与历史重建的前端差异

完整运行报告已经记录本次推理与历史重建结果之间的几何差异：

| 路径 | 当前存在数 | 历史存在数 | 出现性差异 | 几何差异帧 | 最大差异摘要 |
| --- | ---: | ---: | ---: | ---: | --- |
| Base V3 | 992 | 992 | 0 | 10 | center 0.70169 px，long 0.27133 px，short 0.64619 px，angle 0.06480° |
| DINO | 992 | 990 | 2 | 2 | 两帧本次存在而历史缺失；共同存在框差异极小 |
| K1 | 846 | 846 | 0 | 161 | center/angle 为 0，long 最大 0.03441 px，short 最大 0.01311 px |

因此，应表述为“当前冻结入口得到一致的在线观测文件，并完成了历史几何审计”，不能笼统写成所有中间候选与历史文件逐位完全一致。

### 11.3 本地 GT 重算差异

本地使用当前 DOTA 标注解析器重新计算后，与服务器 finalization 中绑定的逐帧值存在以下最大绝对差：

| 指标 | 最大绝对差 | 对应帧 | 方法 |
| --- | ---: | --- | --- |
| 中心误差 | 0.0444445893 px | `real_seq03_00071` | raw |
| 尺度误差 | 0.00000103489105 | `sim_seq09_00257` | raw |
| 角度误差 | 0.0522689819° | `sim_seq09_00326` | raw |
| RIoU | 0.00133404516 | `sim_seq09_00162` | raw |

差异原因尚未完成逐帧核对，可能涉及标注几何转换或环境数值实现，但在得到证据前不作原因归因。

为避免混合来源，自定义指标补充采用服务器 finalization 已绑定的逐帧中心/方向误差和 RIoU，DFR/ACI 使用本次在线框；本地 GT 重算值只作为差异审计保存，没有替换正式统计。

## 12. 正式复现命令

### 12.1 验证历史冻结报告输入

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode validate
```

`validate` 只校验协议和 SHA256，不运行模型。

验证指定的新运行目录时增加 `--input-dir`：

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode validate \
  --input-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_run_v2
```

### 12.2 只运行无 GT 在线推理

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode infer \
  --device cuda:0 \
  --out-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_run_v2
```

该阶段只生成在线 JSON、无 GT 观测 CSV 和 inference receipt。它不读取标注，不生成正确性标签。新目录中已有不同文件时会拒绝覆盖。

### 12.3 对已有在线输出进行 GT 后验评估并生成报告

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode evaluate \
  --input-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_run_v2
```

不提供 `--out-dir` 时，评估结果写入输入运行目录。若要把在线输出和评估归档分开，可增加一个新的 `--out-dir`。该阶段会：

- 核对无 GT 观测 CSV 与在线 JSON 的对应输出。
- 连接固定标注并生成逐帧误差。
- 写入 finalization、论文指标、诊断输入和三份运行时契约。
- 自动生成自定义时序指标。
- 生成模型总报告和独立可靠性报告。

### 12.4 从绑定结果重新生成报告

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode report \
  --input-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_run_v2 \
  --out-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_report_v2
```

该模式不运行模型，也不重新连接 GT。它核对生成结果之间的 SHA256 绑定，并从绑定文件重新生成总报告。省略 `--input-dir` 时读取配置里冻结的历史正式输入。

### 12.5 服务器一键完整运行

完整运行必须使用新的输出目录，避免覆盖历史报告：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode full \
  --device cuda:0 \
  --out-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_full_run_v2
```

`full` 依次执行 infer、evaluate 和 report，自定义指标已经自动进入模型报告。输出目录已经含有不同结果时会拒绝覆盖；不能覆盖 V1 历史产物。

### 12.6 V1 历史下载文件的额外审计

当前补充脚本固定核对 `unified_full_run_v1` 的 9 份 SHA256：

```bash
cd /Users/mac/Documents/paper/symEOOD

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.analyze_unified_full_run_v1
```

该脚本输出：

```text
work_dirs/base_v3_obb_reliability_baseline_v1/unified_full_run_v1/
verified_custom_metrics_v1.json
verified_custom_metrics_v1.md
```

该脚本继续用于固定核对 V1 的 9 份 SHA256、保存本地标注身份并记录本地 GT 重算差异。新 `infer/evaluate/full/report` 流程已经直接调用其中的通用自定义指标实现，无需再运行这个 V1 专用入口才能获得自定义指标。

### 12.7 回归测试

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_base_v3_obb_focused_paper_pipeline_v1.py \
  tests/test_analyze_unified_full_run_v1.py \
  tests/test_eval_crane_offline_records.py
```

当前这三份测试在本地合计 20 项通过，覆盖历史与新运行目录加载、内嵌 SHA256、运行时契约不修改模板、无 GT 推理导出、自定义指标入报告、CSV 重复写入一致性和不同内容拒绝。相关在线及诊断回归合计 45 项通过，全部顶层项目测试为 372 项通过。

`pytest tests` 全树收集在本地 Python 3.13 环境因缺少 `mmcv` 和 `mmdet` 出现 15 个上游 MMRotate 测试导入错误，未进入执行。这不属于本轮断言失败；服务器 Python 3.8 的 `mmrotljj` 环境应补跑受影响测试并记录实际结果。测试数量不能替代真实入口和文件身份校验。

### 12.8 本地与服务器源码、契约比较

服务器执行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

sha256sum \
  crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  crane_project/tools/base_v3_obb_focused_paper_pipeline_v1.py \
  crane_project/tools/analyze_unified_full_run_v1.py \
  crane_project/data_contracts/base_v3_obb_true_online_pipeline_v1.json \
  crane_project/data_contracts/base_v3_obb_true_online_finalization_v2.json \
  crane_project/data_contracts/base_v3_obb_true_online_paper_metrics_v3.json \
  crane_project/data_contracts/base_v3_obb_hybrid_fixed_test_eval_v51.json \
  crane_project/data_contracts/base_v3_obb_hybrid_fixed_test_diagnostic_v51.json
```

本地 macOS 执行同一清单时将 `sha256sum` 换成 `shasum -a 256`。同步前后逐项比较，不放宽协议或哈希校验。

## 13. 论文结果使用建议

当前结果可以支持以下事实性陈述：

- 已形成按时间顺序执行的 RGB 图像到抓斗 OBB 分量观测链。
- 系统可以分别输出中心、尺度和方向，并对每个分量附带有效性及来源状态。
- 高置信度或 `valid=true` 不保证三个几何分量同时正确。
- V5.1 提高了尺度和方向的分量输出覆盖率，但付出了有效输出误差、完整 OBB 覆盖率和联合正确覆盖率方面的代价。
- 仅置信度拒绝提高了保留完整框的平均 RIoU，但因拒绝导致全帧命中覆盖率及连续可用性下降。
- 当前尺度风险排序没有在总体分组的同覆盖率比较中战胜分数排序。
- 当前 1 帧方向保持在全体样本上总体退化，只有部分小分组体现收益。

当前结果不能支持以下结论：

- V5.1 是整体最优方法。
- 分量有效性标志等同于正确性保证或严格校准概率。
- 部分分量可用可以替代完整 OBB 性能。
- fallback 通路已经获得可靠完整 OBB。
- 固定 TEST 的结果代表新未知港口序列的泛化性能。
- 当前总耗时代表纯检测 FPS 或可靠性模块单独开销。
- 当前观测能够直接恢复三维状态、物理摆角或控制变量。

## 14. 后续仍需完成的实验工作

在不根据固定 TEST 调参的前提下，后续论文实验至少还应补齐：

1. 整理检测方法对比和最终保留改进的消融证据，使“抓斗有向检测”部分形成独立完整表格。
2. 在预先冻结协议后使用真正未参与开发的序列验证可靠性策略，或明确把当前研究结论限定为固定 TEST 诊断。
3. 核对本地与服务器 GT 解析及逐帧误差差异，记录 OpenCV、NumPy、标注文件和几何规范身份。
4. 核对当前运行与历史重建中 10 帧 Base V3、2 帧 DINO、161 帧 K1 差异的具体来源。
5. 单独测量前端检测、Base V3 细化、V5.1 判定和报告写入耗时，避免混用总流程时间。
6. 准备真实序列的代表性成功、部分分量可用、连续缺测和失败案例；样本选择规则需预先固定，避免用 GT 挑选有利样本。
7. 若要把可靠性判定写成已成立的主要贡献，需要在独立证据上证明它相对仅置信度筛选具有明确优势；当前固定 TEST 结果只证明了分量级输出机制及其权衡。

## 15. 当前最终状态

模型工程链已经完成：

```text
RGB 图像
  → 冻结 DINO/K1/SymEOOD-Base-V3 检测
  → V5.1 分量判定
  → 带有效性标志的连续观测
  → GT 后验评估
  → JSON、CSV、Markdown 归档
```

结果解释应始终保留以下结论：

> V5.1 不是整体最优方法。它的主要现象是增加部分尺度与方向观测，但这些收益必须与有效输出误差、完整 OBB 覆盖率、联合正确覆盖率和连续缺测代价同时报告。

本文档、原始服务器产物、补充自定义指标报告和状态文件共同构成当前小论文模型与实验阶段的交接证据。后续写作应从这些绑定结果提取数值，不从聊天记录手工抄写近似值。

## 16. 2026-09-16 流程精简修订

本次只优化小论文的评估、报告与核验流程，不训练模型，不修改 DINO、Base V3、V5.1、阈值、checkpoint 或固定 TEST 结果。V5.1 继续作为冻结诊断基线，不作为已经成立的最终可靠性方法，也不建立正式消融入口。

### 16.1 报告分层

统一入口现在生成五份报告文件：

| 文件 | 用途 |
| --- | --- |
| `fixed_test_focused_paper_model_pipeline_v1.json` | 完整机器可读模型、指标和证据链 |
| `fixed_test_focused_paper_model_pipeline_v1.md` | 精简检测与时序主报告 |
| `fixed_test_focused_paper_reliability_v1.json` | 完整机器可读可靠性诊断 |
| `fixed_test_focused_paper_reliability_v1.md` | 只展示“输出率 / 输出准确率”的可靠性主报告 |
| `fixed_test_focused_paper_reliability_detail_v1.md` | 逐分量、同覆盖率、角度保持、来源和连续缺失详细表 |

检测主报告保留 `R_center`、`mean_RIoU`、`DFR`、`ACI`、`TDR_w10`、`MCML_max`，仿真增加 `A-RMSE`。其余自定义指标继续保存在 JSON，不挤入主表。可靠性主报告优先列真实港口结果，再列仿真和全部数据。完整框准确要求中心、尺度、方向同时满足冻结阈值。

### 16.2 GT 一致性审计

新增只读模式 `audit-gt`。它重新解析当前本地 992 份 DOTA 标注，重算逐帧中心、尺度、方向误差和 RIoU，再与绑定的服务器 finalization 比较。该模式只生成差异报告，不改写正式指标。

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode audit-gt \
  --input-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_archive_v2 \
  --out-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_archive_v2_audit
```

输出文件为 `gt_consistency_audit_v1.json`。当前本地复核状态为 `LOCAL_RECOMPUTATION_DIFFERENCES_PRESENT`；最大差异与此前审计一致：中心 `0.0444445893 px`、尺度 `1.03489105e-6`、方向 `0.0522689819°`、RIoU `0.00133404516`。这些差异未用于替换正式结果。

### 16.3 连续状态案例

`visualize` 仍按首次出现的分量状态组合选择案例，不读取 GT 或误差排序；每个案例现在同时输出前后各 2 帧的同序列上下文。manifest 记录中心帧、上下文帧、三分量状态、有效分量和图像 SHA256，可用于制作测量、方向保持和缺测的连续序列示例。

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m crane_project.tools.base_v3_obb_focused_paper_pipeline_v1 \
  --config crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json \
  --mode visualize \
  --input-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/unified_staged_archive_v2 \
  --out-dir \
    work_dirs/base_v3_obb_reliability_baseline_v1/focused_paper_visualization_v2
```

状态颜色只表达输出状态：灰色是 Base V3 框，绿色是完整 V5.1 OBB，黄色是完整框不可用时仍有效的中心。颜色不表示该输出经 GT 证明正确。

### 16.4 后续阶段

当前流程优化完成后，下一阶段优先处理检测前端的 DINO 问题。可靠性性能优化与正式消融暂缓；只有检测基线重新冻结、可靠性方法在独立训练/验证划分上达到预先定义的覆盖率—准确率标准后，才进入可靠性消融和最终 TEST。
