# 创新点 4 后续：统一 DINO 远距离与小目标改进实验总账

更新时间：2026-08-07
项目：SymEOOD / CraneOBB 旋转目标检测
状态：S7 已基本解决小目标候选覆盖，现有 affine/lane/static merge 均已完成并关闭；阶段三
student 通过 source gate 但未改善 `seq03_small` Top-1。native-protected selective promotion V1
已完成服务器 source-only 训练，但四个 epoch 均不发生 S7 接管、未通过 source gate，正式模型
仍为 native S14 alpha=0.5。当前应关闭 V1 的重复调参，同时保留 `seq03_small` 的通用排序问题
为未完成项，随后进入统一训练入口、轻量化和未知序列泛化验证。

## 1. 文档目的

本文记录从冻结 DINOv2 暗光大目标分支取得有效进展后，开始处理远距离和小目标问题以来的全部有用实验。记录范围包括：

- 每项实验要回答的问题；
- 实验修改了什么、冻结了什么；
- source-only 选择协议与 target diagnosis 边界；
- 关键结果、失败原因和是否继续使用；
- 已完成但容易被误认为“还没做”的实验；
- 2024--2026 年相关论文及其与当前瓶颈的准确对应关系；
- 当前阶段性路线、停止门和后续系统闭环顺序。

暗光大目标方法本身见：

```text
docs/innovation4_frozen_dinov2_lowlight_large_target.md
```

本文不会把下面三类困难片段混在一起：

| 切片                     | 主要目标尺度/现象                       | 当前角色                              |
| ------------------------ | --------------------------------------- | ------------------------------------- |
| `real_seq02[137..169]` | 暗光大目标，原 BrightAug 分类排序坍塌   | 已完成 DINO 语义救援，target-dev 诊断 |
| `real_seq02[2..41]`    | 远距离目标，历史 BrightAug 候选不足     | 独立 target diagnosis                 |
| `real_seq03[129..192]` | 更小的旋转目标，短边接近一个 DINO token | 独立 target diagnosis                 |

## 2. 起点：暗光 DINO 已经解决了什么

暗光分支采用冻结 DINOv2 ViT-L/14，只在 source 上训练 oriented RPN 和 rotated ROI head。正式 source-only 模型在 `real_seq02[137..169]` 上取得：

```text
BrightAug：      top1 = 0/33，RIoU-MCML = 33
正式 Scoped DINO：top1 = 29/33，RIoU-MCML = 1
稳定化 DINO：    top1 = 32/33，RIoU-MCML = 1
```

该结果说明冻结 DINO 的语义特征能够绕开 BrightAug 的暗光分类路由失配。但当时的 DINO 检测头只有一个 stride-14 patch-grid 特征层：

```text
Frozen DINOv2 ViT-L/14
  -> final patch tokens, stride 14
  -> single-scale OrientedRPNHead
  -> RotatedRoIAlign, featmap_strides=[14]
  -> RotatedShared2FCBBoxHead
```

因此，小目标阶段的核心问题不是“DINO 是否有语义”，而是这些语义能否以足够密集的空间分辨率形成候选，并在 ROI/NMS 阶段保留正确顺序。

## 3. 数据和评价协议

### 3.1 source-only 训练和选择

```text
source train：train + train_sim，共 2781 帧
source val：  official val，共 738 帧
source-small：按 source token 尺度定义，共 350 帧
```

所有新增训练实验均满足：

```text
DINOv2 frozen = True
target_used_for_training = False
target_used_for_checkpoint_selection = False
```

### 3.2 target 只用于训练完成后的诊断

```text
seq02_far：  test/real_seq02[2..41]，40 帧
seq02_dark： test/real_seq02[137..169]，33 帧
seq03_small：test/real_seq03[129..192]，64 帧
```

这些 target 切片已经参与问题定位和方法设计，不能用于 checkpoint 或超参数选择，也不能称为完全未观察的 final test。正确名称是：

```text
source-trained, target diagnosis-only
```

### 3.3 主要指标

- `top1_hits`：最终第一名检测框满足 `RIoU >= 0.5` 的帧数；
- `top1_mcml`：连续 top-1 失败的最长帧数；
- `RPN R@K`：前 K 个 RPN proposal 中存在 `RIoU >= 0.5` 候选的比例；
- `decoded_geometry_exists`：经过 ROI bbox regression 后是否仍存在可用几何；
- `post-NMS R@K`：NMS 后候选是否保留；
- source exact retention：正式 DINO 原先正确的 source 帧是否全部保留。

## 4. 当前结论摘要

### 4.1 已确认的证据链

1. **当前正式部署基线不变。**
   `work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth` 是唯一部署候选：native S14、ROI classifier `alpha=0.5`、S7 disabled；source full/small=`677/738`、`303/350`、MCML=`3/3`、old-correct lost=`0`。
2. **`seq03_small` 最初确有空间采样瓶颈，但不再是唯一主因。**
   它的短边约 `1.124` DINO token，S14 RPN@2000=`45/64`；anchor 覆盖=`63/64`、assignment=`64/64`，排除了“理论 anchor 不存在”。朴素多尺度可将 RPN 上界提高到 `56/64`，但会伤害 far/dark，故不可部署。
3. **高分辨率 S7 已证明可补足最终候选覆盖。**
   protocol-25 的 source-safe high-resolution ROI readout 在 source 达到 full/small=`688/738`、`311/350`、lost=`0`；固定 target-dev 中，`seq03_small` R@20=`55/64→63/64`、R@100=`55/64→64/64`。
4. **当前 target 失败是最终质量排序，而不是候选缺失。**
   同一 protocol-25 下，far=`38/40`、dark=`29/33`、small=`50/64` 均无 Top-1 增益；small 的 14 个失败帧中，正确候选已位于 rank `2`（9 帧）、`3`（2 帧）、`8/12/39`（各 1 帧）。S7 promotion 仅发生在原本已经正确的 3 帧。
5. **最根本的问题是 source 监督可辨识性与 split 支持错配。**
   现有 source-train 的 real `2,033` 帧全部 native 正确；natural native-wrong/S7-correct gain 只在 `sim_seq08` 有 7 帧。相反，旧 validation 中有 57 个自然 gain（`real_seq07=9`、`sim_seq10=48`），不能直接并入旧训练而不破坏 source gate。protocol-31 的完整图像级 paired-view 只产生 10 个 role-switch，仍全部来自 `sim_seq08`；protocol-32 的 sequence cross-fit 没有任何 viable real held-out fold。
6. **对象级时序可分性不等价于安全的 native--S7 接管质量。**
   protocol-33 正确证明了 source ROI 表征在 GT 正候选锚定下具有跨帧、跨视图对象级可分性；但 protocol-34 的闭环选择发生递归错误状态污染，protocol-35 即使改为 native-only 状态锚定，最佳 source 也仅为 full/small=`663/738`、`290/350`、lost=`31`，仍低于 native baseline。因此，时序身份一致性不能直接充当候选质量或 takeover criterion。

### 4.2 根本问题的准确表述

> 在 `seq03_small` 所代表的未知小目标场景中，正确 S7 候选已经能够进入最终候选池，但现有 source 数据**没有同时提供**跨 real/sim、跨 sequence 的“native 失败而 S7 正确”训练证据，以及独立 real hard sequence 的留出验证证据。因此，当前无法从既有 source split 中识别、训练并证明一个既能提升正确候选、又保持 native exact-retention 的统一质量排序/接管器。

这一定义区分了三件事：`seq03_small` 是固定 target-dev 上的**症状与诊断载体**，不是训练样本；S7 候选覆盖是已取得的结构性进展；未解决的是安全跨域排序所需的 source supervision，而不是继续扫描 margin、扩大 S7 RPN 或叠加同类后处理。

### 4.3 当前允许与禁止的动作

- 允许：冻结当前证据；补充至少两个独立标注的真实 source 困难连续序列；按完整 sequence 预注册训练/验证 split 后，再训练统一 quality head。
- 禁止：把 `real_seq07` 或 `sim_seq10` 直接塞回旧训练集；使用 target、target GT、target-derived threshold 或 `seq03_small` 逐帧规则；重复 target-dev、margin scan、同构 S7 selector 训练。
- 即使未来 source gate 通过，固定 target-dev 也只是一轮诊断；真正泛化仍需未参与设计的新连续小目标序列。GTX 1080 的 DINO teacher 延迟约 `1.74--1.98 s/frame`，应在有效统一结构确认后再做蒸馏/加速。

## 5. 实验时间线和详细结论

本节仅保留复现实验、追溯 checkpoint 与核对历史差异所需的细节；当前研究判断以第 4 节的
证据链和第 6 节的停止裁决为准。历史 target 数字只能在同一方法、checkpoint 和 gate 下比较，
不得跨实验拼接为单一模型结果。

### 5.1 远距离候选生成初筛

工具：

```text
crane_project/tools/dino_teacher_far_distance_candidate_audit.py
```

目标：检查原冻结 DINO labeller 的最终输出能否覆盖 `real_seq02[2..41]`。

早期日志：

```text
geometry = 0/40
R@20 = 0
R@100 = 0
top1 = 0
decision = DINO_FAR_DISTANCE_CANDIDATE_GENERATION_INSUFFICIENT
```

解释：这个初筛检查的是当时特定候选输出路径的最终可用结果，不等价于逐级 RPN proposal 覆盖。后续 RPN-to-ROI attrition 审计证明正式 DINO 的 `seq02_far` RPN 实际为 `40/40`，因此不能继续引用 `0/40` 作为“DINO RPN 完全没有远距候选”的结论。

状态：**被后续更细的逐级审计取代，仅保留为实验历史。**

### 5.2 Token-Scale / RPN Coverage Audit

工具和结果：

```text
crane_project/tools/dino_teacher_token_scale_rpn_coverage_audit.py
token_scale_rpn_coverage_audit_v1.json
token_scale_rpn_coverage_audit.json
token_scale_rpn_coverage_selected.json
```

目标：区分以下三种解释：

1. 目标比 DINO patch 还小，空间采样不足；
2. anchor 本身没有几何覆盖；
3. anchor 可覆盖，但训练后的 RPN 未把正确 proposal 排进候选池。

该审计是严格只读的：`optimizer_steps=0`、`checkpoint_writes=0`、DINO/RPN/ROI 参数均未变化。

关键结果：

| 数据            | 短边 token 中位数 | anchor@0.5 | RPN@20 | RPN@2000 |
| --------------- | ----------------: | ---------: | -----: | -------: |
| source-small    |                 - |   约 0.969 |  0.589 |    0.726 |
| `seq02_far`   |             1.498 |      1.000 |  0.900 |    1.000 |
| `seq02_dark`  |             2.683 |      1.000 |  0.788 |    1.000 |
| `seq03_small` |             1.124 |      0.984 |  0.594 |    0.703 |

三个 checkpoint 的 RPN 数字相同：正式 DINO、small-hard ROI 分类头和 source-safe 插值头都没有修改 RPN。这是正确现象，不是缓存错误。

结论：

```text
seq03_small 的理论 anchor 基本存在；
真正不足的是 stride-14 DINO 特征上的已学习 RPN objectness/regression；
单纯继续训练 ROI 分类器不能提高 RPN@K。
```

状态：**有效归因，作为下一结构设计的核心证据。**

### 5.3 RPN-to-ROI Attrition + Latency Audit V1/V2

工具：

```text
crane_project/tools/dino_teacher_rpn_roi_attrition_latency_audit.py
rpn_roi_attrition_latency_audit_v1.json
rpn_roi_attrition_latency_audit_v2.json
rpn_roi_attrition_selected_v2.json
```

目标：逐级定位候选从 RPN 到最终 top-1 的损失位置：

```text
RPN proposal
  -> ROI regression
  -> decoded geometry
  -> rotated NMS
  -> valid-content filter
  -> final top-1 ordering
```

V1 曾存在归因口径不完整的问题，V2 增加了 NMS/ordering/regression 的互斥终端原因和一致计数。后续结论以 V2 为准。

使用 `source_safe_interpolated_head.pth` 的 V2 结果：

| 切片            | RPN recall | decoded geometry | final top1 | MCML | 终端失败                            |
| --------------- | ---------: | ---------------: | ---------: | ---: | ----------------------------------- |
| `seq02_far`   |      1.000 |            1.000 |      38/40 |    1 | 2 个 ordering/NMS                   |
| `seq02_dark`  |      1.000 |            1.000 |      29/33 |    1 | 4 个 ordering/NMS                   |
| `seq03_small` |      0.703 |            0.984 |      50/64 |    6 | 13 个 ordering/NMS，1 个 regression |

这里 `seq03_small` 的 decoded geometry 高于 RPN recall 并不矛盾：部分初始 proposal 的 RIoU 小于 0.5，但 ROI regression 能把它们修正为可用旋转框。

可靠结论：

```text
seq03_small 不是单一 RPN miss；
它是 stride-14 候选覆盖不足 + ROI ordering/NMS 的复合问题；
但 regression 只造成 1/14 个终端失败，不是第一优先级。
```

延迟和参数量：

```text
Frozen DINOv2：304,368,640 parameters
RPN + ROI heads：54,824,560 parameters
source DINO branch p50：约 1.740 s/frame
seq03_small p50：约 1.975 s/frame
```

测试设备为三张 GTX 1080 8G，DINO blocks 分片运行。主要耗时来自冻结 DINOv2 前向，而不是 RPN/NMS。

状态：**有效归因和实时性基线。**

### 5.4 ROI Classification Small-Hard 训练

训练结果：

```text
本地结果：train_result.json
server work_dir：dino_teacher_roi_cls_small_hard_v1
trainable：仅 fc_cls.weight / fc_cls.bias，共 2050 参数
epochs：4
```

目标：在不修改 DINO、RPN、ROI regression 的情况下，用 source-small hard samples 改善 ROI 分类排序。

source-val：

| 模型            | full top1 / MCML | small top1 / MCML |
| --------------- | ---------------: | ----------------: |
| 正式 DINO       |      662/738 / 7 |       293/350 / 7 |
| small-hard best |      679/738 / 4 |       304/350 / 4 |

target V2 attrition：

```text
seq02_far：39/40
seq02_dark：29/33
seq03_small：47/64
```

结论：source 指标明显改善，但直接使用完整 small-hard 分类器并没有在 `seq03_small` 上稳定迁移；只看 source top1 会掩盖旧正确帧被替换和 target 排序变化。

状态：**不直接部署，但保留为可插值的新分类器端点。**

### 5.5 Pairwise Ranking + Retention V1

训练结果：

```text
本地结果：Copy of train_result.json
server work_dir：dino_teacher_roi_cls_pairwise_retain_v1
trainable：仅 ROI fc_cls，共 2050 参数
```

目标：对每个 source 正样本与 hard negative 建立 pairwise margin，同时用原分类器 logits 作为 retention teacher，直接学习“正确 ROI 应排在相似错误 ROI 前面”。

4 个 epoch 都改善 source top1，但没有满足严格的 exact retention，因此正式 selector 回退到 epoch 0：

| epoch | full top1 / MCML | small top1 / MCML | exact retention |
| ----: | ---------------: | ----------------: | --------------- |
|     1 |          678 / 4 |           302 / 4 | FAIL            |
|     2 |          677 / 4 |           306 / 4 | FAIL            |
|     3 |          679 / 4 |           307 / 4 | FAIL            |
|     4 |          677 / 4 |           306 / 4 | FAIL            |

随后做了 source-only epoch 重选，允许最低保留率 `0.985`，选出 epoch 1：

```text
retention = 0.986405
full = 678/738
small = 302/350
checkpoint = labeller_best_source_retained_985.pth
```

三切片 attrition：

```text
seq02_far： 39/40，MCML=1
seq02_dark：29/33，MCML=1
seq03_small：45/64，MCML=6
```

结论：Pairwise V1 对 far slice 有利，但没有解决 `seq03_small`，并且 exact retention 未通过。

状态：**不作为最终 checkpoint；其分类方向只用于后续安全权重插值。**

### 5.6 Source-Safe FC Classifier Weight Interpolation

工具和结果：

```text
crane_project/tools/dino_teacher_fc_cls_interpolation_selector.py
source_interpolation_result.json
work_dirs/dino_teacher_fc_cls_interpolation_v1/
source_safe_interpolated_head.pth
```

目标：在正式分类器与更新后分类器之间插值，只使用 source-val 选择最大安全更新幅度：

```text
W(alpha) = (1-alpha) * W_formal + alpha * W_updated
```

这是一项**分类器权重插值**，不是 DINO 空间特征上采样。

结果：

|           alpha |  full top1 / MCML | small top1 / MCML | 旧正确帧丢失 | Gate                      |
| --------------: | ----------------: | ----------------: | -----------: | ------------------------- |
|           0.000 |           662 / 7 |           293 / 7 |            0 | baseline                  |
|           0.125 |           666 / 5 |           297 / 5 |            0 | PASS                      |
|           0.250 |           671 / 4 |           300 / 4 |            0 | PASS                      |
| **0.500** | **677 / 3** | **303 / 3** |  **0** | **PASS / selected** |
|           0.750 |           678 / 3 |           303 / 3 |            3 | FAIL                      |
|           1.000 |           679 / 4 |           304 / 4 |           10 | FAIL                      |

`alpha=0.5` 完整保留原先 662 个正确 source 帧，并新增 15 个正确帧，是当前最强且满足 exact retention 的分类器 checkpoint。

状态：**已完成、有效、当前小目标研究的分类头基线。不要再次重复权重插值。**

### 5.7 Rotated-NMS Retention Audit

工具和结果：

```text
crane_project/tools/dino_teacher_rotated_nms_retention_audit.py
nms_retention_audit_v1.json
```

目标：判断原 DINO ROI `nms_iou_thr=0.1` 是否过强地抑制小目标附近的正确框，并只用 source-val 选择 NMS 策略。

source-only 选择结果：

```text
selected NMS IoU = 0.5
source top1 = 662/738
source exact retention = 662/662
source-small post-NMS R@20 = 0.920
```

target diagnosis：

```text
seq02_far： 38/40，MCML=1，post-NMS R@20=0.975
seq02_dark：29/33，MCML=1，post-NMS R@20=0.970
seq03_small：52/64，MCML=6，post-NMS R@20=0.875
```

结论：`0.5` 是 source-selected 的更合理 DINO ROI NMS 阈值，能够保留更多正确 ROI，但它仍是推理策略，不是空间表征创新，也没有让 MCML 从 6 降下来。

该 NMS 只属于 DINO ROI head，不改变原 `crane_symeood_k1.py` / BrightAug 的无 NMS 设计，因此不存在结构冲突。

状态：**保留为后续统一 DINO 的固定推理设置和候选接口，不单独作为论文创新。**

### 5.8 插值式多尺度 RPN Coverage Audit

配置和结果：

```text
crane_project/configs/crane_symeood_scoped_dino_lowlight_multiscale_v1.py
multiscale_rpn_coverage_v1.json
feature_strides = [7, 14, 28]
```

目标：不训练新参数，仅把同一个 stride-14 DINO patch grid 双线性插值/池化为 stride 7、14、28，检查多尺度候选覆盖上界。

该实验：

```text
optimizer_steps = 0
audit_trainable_parameters = 0
使用原正式 DINO head 权重
```

与单尺度对比：

| 数据            | 单尺度 RPN@20 |   多尺度 RPN@20 | 单尺度 RPN@2000 | 多尺度 RPN@2000 |
| --------------- | ------------: | --------------: | --------------: | --------------: |
| source          |         0.713 |           0.710 |           0.835 |           0.874 |
| `seq02_far`   |         0.900 |           0.800 |           1.000 |           0.950 |
| `seq02_dark`  |         0.788 |           0.636 |           1.000 |           0.909 |
| `seq03_small` |         0.594 | **0.719** |           0.703 | **0.875** |

`seq03_small` RPN@2000 从 `45/64` 提高到 `56/64`，证明提高空间分辨率是有效方向。但朴素共享权重的三尺度同时损伤 far 和 dark，说明不同尺度不能只靠固定插值和同一 RPN 权重机械处理。

状态：**有效上界和方向证据；当前 config 仍是实验配置，不能作为正式模型。**

### 5.9 Pairwise Ranking V2

训练结果：

```text
本地结果：Copy of Copy of train_result.json
server work_dir：dino_teacher_roi_cls_pairwise_v2_formal4_v1
epochs：4
```

目标：只在 source 中确实发生 top-1/NMS 排序失败的帧上构造 paired positive/negative，并要求 exact source retention。

结果：

| epoch | full top1 / MCML | small top1 / MCML | 旧正确帧丢失 | pair signal |
| ----: | ---------------: | ----------------: | -----------: | ----------- |
|     1 |          674 / 3 |           301 / 3 |            3 | 极少        |
|     2 |          672 / 5 |           299 / 5 |            2 | 极少        |
|     3 |          670 / 5 |           299 / 5 |            2 | 0           |
|     4 |          670 / 5 |           299 / 5 |            2 | 0           |

selector 因 exact retention 失败而回退 epoch 0。该结果也被旧 `alpha=0.5` 安全插值严格支配：

```text
旧安全插值：677/738，303/350，MCML=3，丢失0帧
V2 epoch1：674/738，301/350，MCML=3，丢失3帧
```

状态：**关闭。不要对 V2 再做一轮分类权重插值。**

### 5.10 LiDeRe-inspired Residual S7 Dense Readout 训练

工具、配置和结果：

```text
crane_project/tools/dino_teacher_rotated_labeller.py
crane_project/configs/crane_symeood_scoped_dino_lowlight_s7_v1.py
work_dirs/dino_teacher_s7_residual_v1/train_result.json
```

目标：验证一个受 LiDeRe 轻量冻结主干 dense readout 启发的高分辨率补充分支，能否提高
`seq03_small` 所代表的小目标候选能力，同时不破坏已经有效的 S14 DINO 路径。

结构：

```text
Frozen DINOv2 S14 feature
  |-- frozen native S14 RPN
  `-- 1x1 1024->128 -> 2x bilinear upsample
       -> depthwise 3x3 + GroupNorm + GELU + pointwise 1x1
       -> zero-initialized residual gate
       -> trainable S7 Oriented RPN

native S14 proposals + bounded S7 proposals
  -> one shared native ROI head
```

冻结和训练边界：

```text
冻结：DINOv2、原 S14 RPN、ROI classifier alpha=0.5、ROI bbox regression
训练：S7 readout + S7 RPN
target：未读取，未参与训练和 checkpoint 选择
```

source baseline 是 source-safe `alpha=0.5` checkpoint：

```text
full       = 677/738，MCML=3
source-small = 303/350，MCML=3
旧正确帧   = 677
```

训练过程中的结果：

| epoch | mean loss |  full top1 / MCML | small top1 / MCML | 保留旧正确帧 | gate           |
| ----: | --------: | ----------------: | ----------------: | -----------: | -------------- |
|     0 |         - |           677 / 3 |           303 / 3 |      677/677 | PASS，回退基线 |
|     1 |   0.14666 |          660 / 11 |          287 / 11 |      651/677 | FAIL           |
|     2 |   0.02276 | **688 / 3** | **318 / 3** |      654/677 | FAIL           |
|     3 |   0.01965 |           684 / 2 |           312 / 2 |      653/677 | FAIL           |
|     4 |   0.01942 |           685 / 2 |           312 / 2 |      655/677 | FAIL           |

其他记录：

```text
S7 trainable parameters = 309,994
DINO parameters unchanged = True
head peak allocated memory ~= 346 MB
all epoch feature-cache hits = 2781/2781
target_used_for_training = False
target_used_for_checkpoint_selection = False
```

解释：epoch 2 的小目标 source top-1 从 `303/350` 提高到 `318/350`，说明 S7 readout/RPN
确实学到了额外候选；但同时丢失了 23 个原本正确的 source 帧。新增 S7 候选进入同一个
ROI/NMS 排序后抢走了部分 native S14 正确候选。因此这是**方向上的正证据、部署门控上的失败**，
不是训练崩溃，也不是 LiDeRe readout 已被证明无效。

当前裁决：

```text
S7 first implementation = source candidate gain, exact-retention failure
best_epoch = 0
selected checkpoint = native alpha=0.5 behavior with S7 disabled
target test = not run and not authorized
```

当前模块没有实现 LiDeRe 的完整 readout/attention 结构，只借鉴“冻结大 backbone + 轻量
dense readout”原则，因此不能称为 LiDeRe 复现。该实验也不能在论文中报告为最终小目标
改进，只能报告为 LiDeRe-inspired readout 的第一版消融和失败原因。

## 6. 哪些内容已经做过，不能重复

| 证据类别 | 已保留的结论 | 当前裁决 |
|---|---|---|
| 安全部署与固定推理 | FC interpolation `alpha=0.5`、rotated NMS IoU=`0.5` 已由 source exact-retention 选择 | 保留为唯一正式部署基线 |
| 空间采样/候选覆盖 | anchor 并非主因；S7/high-resolution 可提高小目标候选覆盖，但朴素 `[7,14,28]` 多尺度会伤害 far/dark | 保留“高分辨率候选可行”，关闭朴素多尺度 |
| 高分辨率 target 诊断 | protocol-25 将 small R@100=`55→64/64`，但 Top-1 仍 `50/64` | 覆盖问题完成；不能声称检测或泛化完成 |
| 直接排序/merge/promotion | ROI small-hard、pairwise、affine、lane boost/replay、static ranker 均不能同时获得收益与 exact retention | 不再重复调 loss、margin、boost 或同构 ranker |
| 时序/学生统一路线 | source-safe 时序与 candidate student 可维持覆盖，但 fixed small Top-1 未提高 | 不把 temporal far gain 与其他模型拼接；该路线不再解决当前核心问题 |
| 风险抑制与支持审计 | non-positive risk、relative-risk、smooth geometry、paired-view 均未给出跨域可训练正支持 | 关闭在既有 split 上继续训练 selector |
| sequence cross-fit 审计 | 没有 viable real hard held-out fold | 现有 source 不能形成无泄漏跨域 selector 协议 |
| 工程后续 | 统一入口、self-contained checkpoint、蒸馏/加速尚未完成 | 等获得有效且可验证的统一结构后再推进 |

## 7. 论文映射

### 7.1 直接作为当前方法协议依据

#### DINO Teacher，CVPR 2025

论文：[Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Lavoie_Large_Self-Supervised_Models_Bridge_the_Gap_in_Domain_Adaptive_Object_CVPR_2025_paper.html)

可借鉴内容：冻结 DINOv2 backbone，只用 source 标注训练下游 labeller；target 不参与 labeller checkpoint 选择。

本项目已实现的部分：冻结 DINOv2 + source-only oriented RPN/ROI labeller。

未复现的部分：target pseudo-label student、完整 feature-alignment student。论文中不能把当前方法称为 DINO Teacher 全流程复现。

### 7.2 S7 结构依据与当前限制

#### LiDeRe，CVPR 2026

论文：[LiDeRe: A Lightweight Readout for Fast and Data-Efficient Dense Prediction](https://openaccess.thecvf.com/content/CVPR2026/html/Luddecke_LiDeRe_A_Lightweight_Readout_for_Fast_and_Data-Efficient_Dense_Prediction_CVPR_2026_paper.html)

可借鉴内容：在冻结大型视觉 backbone 上训练轻量 dense readout，通过插值和局部细节建模恢复细粒度空间预测，覆盖 object detection 等任务。

与当前证据的对应：`seq03_small` 只有约 1.124 个短边 token；朴素 stride-7 插值曾把 RPN@2000 提高 11 帧，后续 S7 readout/merge 已进一步将固定 target-dev 的 R@100/geometry 提高到 `64/64`。当前不再缺候选，完整 LiDeRe readout 升级应推迟到 source-safe 合并成立之后。

定位：**S7 结构方向依据；第一版候选覆盖已获正证据，但 affine/lane 合并尚未通过保留门控。**

#### Real-Time Object Detection Meets DINOv3，2025

论文：[arXiv:2509.20787](https://arxiv.org/abs/2509.20787)

可借鉴内容：Spatial Tuning Adapter 将单尺度 DINO 特征转换为多尺度细节特征，并强调实时检测的性能/成本折中。

限制：使用 DINOv3 和 DETR/DEIM 系列，不是当前 DINOv2 + rotated RPN/ROI 的直接代码模板。可借鉴 adapter 的轻量多尺度思想，不应整体替换现有检测器。

定位：**LiDeRe 之后的结构细节参考。**

### 7.3 高分辨率特征和小目标备选依据

#### FeatUp，ICLR 2024

论文：[FeatUp: A Model-Agnostic Framework for Features at Any Resolution](https://openreview.net/forum?id=GkJiNn2QDF)

可借鉴内容：学习式上采样低分辨率 backbone 特征，提高小目标的高分辨率特征定位能力。

限制：训练和显存成本可能高于当前所需的 2x stride-7 readout。只有轻量 readout 仍不足时才考虑。

#### ESOD，2024

论文：[ESOD: Efficient Small Object Detection on High-Resolution Images](https://arxiv.org/abs/2407.16424)

可借鉴内容：复用原 backbone 做 feature-level object seeking，只对稀疏目标区域切高分辨率 patch，避免整图放大成本。

与本项目的关系：如果全图 P7 readout 无法满足 GTX 1080 实时预算，可将其作为稀疏高分辨率候选分支；仍应复用同一 DINO，而不是新增独立小目标模型。

#### DCFL / AI-TOD-R，2024

论文：[Oriented Tiny Object Detection: A Dataset, Benchmark, and Dynamic Unbiased Learning](https://arxiv.org/abs/2412.11582)

可借鉴内容：旋转微小目标的动态 prior 和 coarse-to-fine assignment，缓解极小目标正样本数量与质量失衡。

与本项目的关系：当前理论 anchor 覆盖已经很高，因此 DCFL 不应先于 dense readout。只有新 P7 特征形成后仍出现 assignment/objectness 失配，才引入其动态先验或 assignment 思想。

### 7.4 只作旁证，不是当前实现模板

#### RoMa，CVPR 2024

论文：[RoMa: Robust Dense Feature Matching](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html)

旁证：冻结 DINOv2 提供稳健但粗糙的语义特征，专门的 ConvNet fine features 可以补充精确定位。

限制：它是 dense matching，不是旋转目标检测；新增 fine encoder 也会增加模型复杂度。当前先尝试同一 DINO 上的轻量 readout。

#### Guided Distillation，WACV 2024

论文：[Guided Distillation for Semi-Supervised Instance Segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Berrada_Guided_Distillation_for_Semi-Supervised_Instance_Segmentation_WACV_2024_paper.html)

可借鉴的是冻结/预训练 backbone 下的下游检测与分割工程经验。其核心贡献是半监督 guided burn-in，不直接解决本项目的 stride-14 旋转小目标 RPN 问题，因此不作为下一步主依据。

#### Object-DINO，2026

论文：[Finding Distributed Object-Centric Properties in Self-Supervised Transformers](https://arxiv.org/abs/2603.26127)

可借鉴内容：对象信息分布在 Transformer 多层和 q/k/v patch interaction 中，而不只在最后一层 token。

限制：原方法面向 training-free object discovery 和视觉语言 hallucination；需要访问全层注意力并聚类 heads，推理成本和当前 rotated RPN/ROI 接口跨度都很大。

定位：**暂缓，不与 LiDeRe readout 同时实现。**

## 8. 历史阶段建议：non-positive S7 quality suppression

> 本节记录 2026-08-01 当时的下一步及其实现边界，后续实验已经完成并取代该建议。
> 当前有效路线以第 10.17 节为准，不能再把本节命令当作“尚未运行”的任务。

不要重复训练当前 S7，也不要重复运行已经完成的多尺度或 target 审计。全局 affine 和
两版正向 lane arbitration 都已执行并失败；下一步只允许对固定 affine epoch 1 增加
source-only 的非正值质量抑制：

```text
Non-positive S7 quality suppression for Frozen DINOv2
```

S7 readout 本身保留，新增 source-only 的 native-retention 机制。不是重新训练第二个 DINO，
也不是人工判断图像属于暗光还是小目标。所有图像统一经过同一个 DINO。

### 8.1 已完成的第一阶段结构

```text
Frozen DINOv2 P14 tokens
  |-- 原 P14 RPN 路径：权重和 proposal 配额完全保留
  `-- 2x spatial upsample to P7
       -> depthwise refinement（当前实现没有额外 attention）
       -> zero-initialized residual gate
       -> 独立轻量 P7 RPN

P14 proposals + P7 proposals
  -> 保留式 merge
  -> 现有 alpha=0.5 source-safe ROI classifier
  -> DINO ROI NMS IoU=0.5
```

只增加 P7，不先增加 P28。原因是当前问题是小目标空间采样，P28 对该问题没有直接帮助，而且三尺度只读审计已经显示共享多层会伤害 far/dark。

### 8.2 已完成的第一阶段训练边界

先冻结：

```text
DINOv2
原 P14 RPN
ROI classifier alpha=0.5
ROI bbox regression
```

只训练：

```text
P7 dense readout
P7 RPN objectness/regression
```

这一轮已经回答：新增高分辨率 readout 能补充小目标 proposal，但当前合并方式会破坏部分原 S14 排序。

### 8.3 source-only 选择门槛

建议固定为：

```text
当前 source-safe alpha=0.5 初始 checkpoint 的正确帧保留率 = 100%
source full top1 >= 677/738
source-small top1 >= 303/350
source MCML <= 3
候选 epoch 的 source-small 选择键必须优于初始 checkpoint
```

选择键按 `top1 -> R@20 -> R@100 -> mean RIoU` 的顺序比较，因此允许
top-1 持平但 proposal 排名改善。这些门槛使用 source-val；target 不用于选择
readout 结构、epoch、阈值或 checkpoint。RPN@2000 是 checkpoint 固定后的归因
指标，不作为当前训练脚本的直接选择门槛。

### 8.4 下一版 source-only 选择目标

```text
native S14 旧正确帧保留率 = 100%
source full 不低于 677/738
source-small 不低于 303/350
S7 带来的 source-small R@20/R@100 或 mean RIoU 必须有净改善
S7 proposal 只能在 source-calibrated retention 约束下参与最终排序
```

候选需要记录来源 `native_s14` 或 `supplement_s7`。训练时使用 native S14
正确排序作为 retention teacher；推理时仍使用同一个 ROI head，不增加第二个模型。
只有 source gate 通过后，才重新做三个困难切片的 target diagnosis；在此之前不重复 target 测试。

### 8.5 2026-07-30 第一版实现状态

代码中统一改名为 `S7`，避免与标准 FPN 中表示粗尺度的 `P7` 混淆。

当前已经实现：

```text
native S14 DINO feature
  |-- 原 S14 RPN：冻结并完整保留候选配额
  `-- 1x1 1024->128（先降维）
       -> 2x bilinear upsample
       -> depthwise 3x3 + GroupNorm + GELU + pointwise 1x1
       -> zero-initialized residual gate
       -> 独立 S7 Oriented RPN

S14 proposals 全部保留 + 有界 S7 supplement
  -> 原生 S14 ROIAlign
  -> 已验证的 ROI classifier / bbox regressor
```

实现边界：

- `DINOv2` 始终冻结；
- 原 S14 RPN、ROI classifier、ROI bbox regressor 始终冻结；
- 不建立第二套 ROI Head；
- S7 默认关闭，因此旧正式 config 和旧 checkpoint 行为不变；
- 旧 source-safe checkpoint 只允许作为 S14 初始化；恢复 S7 checkpoint 时要求结构严格匹配；
- source baseline 评估时关闭 S7，epoch 1 开始才启用 S7；如果所有 epoch 都未通过 source 保留门，best checkpoint 自动回退到 S7 关闭的 epoch 0；
- 每个 epoch 记录 head GPU 的 peak allocated memory；
- 本阶段不把两卡 BrightAug 训练和三卡 DINO 分支训练封装为一个总 launcher。等小目标结构最终固定后再统一完整训练流程。

第一版 source-only 训练命令（已完成，不要重复运行）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_rotated_labeller.py \
  --data-root crane_project/data/crane_grab/ \
  --source-train-datasets train:train train_sim:train \
  --source-val-datasets val:val \
  --dinov2-repo third_party/dinov2 \
  --dinov2-checkpoint pretrained/dinov2_vitl14_pretrain.pth \
  --dinov2-model dinov2_vitl14 \
  --dino-gpus 1 2 \
  --head-gpu 0 \
  --legacy-sdpa-query-chunk 512 \
  --dino-height 600 \
  --dino-max-long-side 1333 \
  --patch-size 14 \
  --rpn-feat-channels 256 \
  --roi-fc-channels 1024 \
  --roi-samples 256 \
  --proposal-count 2000 \
  --max-detections 2000 \
  --roi-nms-iou-thr 0.5 \
  --s7-residual \
  --s7-channels 128 \
  --s7-rpn-feat-channels 128 \
  --s7-proposal-count 500 \
  --s7-nms-pre 2000 \
  --s7-anchor-sizes 16 32 64 128 256 \
  --s7-source-min-full-top1 677 \
  --s7-source-min-small-top1 303 \
  --s7-source-max-mcml 3 \
  --train-components s7_rpn \
  --source-small-repeat 1 \
  --epochs 4 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --max-grad-norm 10 \
  --warmup-iters 1000 \
  --warmup-ratio 0.001 \
  --lr-steps 2 3 \
  --selection-epochs 1 2 3 4 \
  --checkpoint-interval 1 \
  --init-checkpoint \
    work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth \
  --feature-cache-dir \
    work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache \
  --work-dir work_dirs/dino_teacher_s7_residual_v1 \
  --skip-target-eval \
  --seed 0 \
  --out-json work_dirs/dino_teacher_s7_residual_v1/train_result.json
```

这条命令在物理上使用三张 1080，但 DINOv2 只分片在逻辑 GPU 1/2；逻辑 GPU 0 只承载可训练 readout/RPN 和被冻结的原 head。GPU 数量只用于模型分片，不使用 DDP，也不改变有效 batch size。

### 8.6 2026-07-31 retention-aware merge 代码状态

代码和第一轮服务器 source-only 训练均已完成；target 仍未读取：

```text
正式 native 初始化：source_safe_interpolated_head.pth
冻结 S7 组件来源：  labeller_epoch_02_source_only.pth
S7 checkpoint 加载：只加载 s7_readout.* / s7_rpn_head.*
训练参数：           s7_score_calibrator.raw_scale / bias，共 2 个标量
候选来源：           native_s14 / supplement_s7
合并位置：           shared ROI 解码后、最终排序前
NMS：                两个来源分别执行，S7 不删除 native 候选
target：             source gate 通过前不读取
```

服务器结果：

| epoch | full top1 / MCML | small top1 / MCML | 保留旧正确帧 | S7 top1 帧 | gate |
| ----: | ----------------: | ----------------: | -----------: | ---------: | ----: |
| 0 | 677 / 3 | 303 / 3 | 677/677 | 0 | PASS，回退基线 |
| 1 | 688 / 3 | 311 / 3 | 676/677 | 42 | FAIL |
| 2 | 687 / 3 | 311 / 3 | 675/677 | 54 | FAIL |
| 3 | 686 / 3 | 310 / 3 | 674/677 | 55 | FAIL |
| 4 | 686 / 3 | 310 / 3 | 674/677 | 55 | FAIL |

候选覆盖已明显改善：full `R@100 717 -> 738`，source-small `329 -> 350`。
但训练集中约 `2774/2781` 个 retention pair 的 hinge 从第一轮开始就是 0，实际梯度
主要来自约 7 个 gain pair。全局 affine 因此持续抬高 S7 分数，增益帧固定为 12，丢失帧
却从 1 增加到 3。该结果拒绝的是两标量全局校准，不是否定 S7 候选或 LiDeRe-inspired
readout。

2026-07-31 诊断代码补充：每个 epoch 记录 affine scale/bias、retention/gain active
constraint 数量，并将 lost/gained 帧的 native/S7 分路 top-1 score、RIoU、来源和
pre-NMS log-odds 写入 `source.history[].s7_merge_conflicts`。这只增强 source-only
可观测性，不改变模型输出、训练损失或 gate。

同时增加 `--source-conflict-result-json` / `--source-conflict-epoch` 只读入口。它与
`--eval-only-checkpoint --skip-target-eval` 联用，直接从既有 JSON 读取 changed frame
keys，只复核 epoch 1 的 1 lost + 12 gained source 帧，不重训、不遍历完整 source-val、
不读取 target。

该审计实际输出为：

```text
[source-val] 13/13 top1_hits=12
decision = SOURCE_ONLY_CONFLICT_AUDIT_COMPLETE_TARGET_NOT_READ
source val audited = 13 changed frames
top1 correct = 12/13
mode = eval-only, no retraining
target_dev = null
```

这里的 `13` 只包含 epoch 1 相对 native 改变的 `1 lost + 12 gained` 帧，不是重新评估
全部 738 帧；`12/13` 与全量 source-val 的 `676/677` retention、`+12/-1` 完全一致。

冲突审计后采用独立的宽松 target-dev 诊断门，正式 exact-retention selector 不变：

```text
retained >= 676/677
full top1 >= 685/738
small top1 >= 308/350
full/small MCML <= 3
gained/lost >= 5
```

epoch 1 满足该门，只被授权作为一次 target-dev 诊断候选，不具备部署资格。新增工具
`crane_project/tools/dino_teacher_s7_relaxed_target_audit.py` 在同一份冻结 DINO 特征上
对比 native alpha=0.5 与固定 epoch 1，一次运行固定的 `seq02_far`、`seq02_dark`、
`seq03_small`。far/dark 要求 top-1 与 MCML 不回退；seq03_small 要求 top-1 严格提升
且 MCML 不回退。输出逐帧 gained/lost，target 不参与 checkpoint 或参数选择。

新训练模式和部署配置：

```text
--train-components s7_merge
crane_project/configs/
  crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py
```

训练使用 source GT 无梯度挖掘两类 pair：native 正确时约束错误 S7 候选不能
覆盖 native；native 错误且 S7 存在 usable candidate 时学习安全增益。RIoU 和 NMS
只参与 pair mining，不直接作为损失。checkpoint 仍必须通过 `677/677` exact
retention、full/small top-1 和 MCML gate；如果没有候选 epoch 通过，继续回退到
S7-disabled epoch 0。

### 8.7 2026-07-31 三切片 relaxed-gate target-dev 结果

固定 native alpha=0.5 与 epoch 1 S7 merge，在同一份冻结 DINO 特征上完成一次配对
target-dev 审计，输出为 `work_dirs/dino_teacher_s7_relaxed_target_audit_v1/result.json`。
该结果没有训练、没有调参，也没有使用 target 选择 checkpoint：

```text
decision = S7_RELAXED_TARGET_DEV_DIAGNOSTIC_PASS
eligible_for_deployment = false
eligible_for_final_test = false
eligible_for_next_stage = true
optimizer_steps = 0
```

| 固定切片 | native top-1 | S7 merge top-1 | MCML | usable geometry | R@20 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_seq02[2..41]` far | 38/40 | 38/40 | 1 -> 1 | 无变化 | 40 -> 40 | 40 -> 40 |
| `real_seq02[137..169]` dark | 29/33 | 29/33 | 1 -> 1 | 无变化 | 33 -> 33 | 33 -> 33 |
| `real_seq03[129..192]` small | 50/64 | 51/64 | 6 -> 6 | 55 -> 64 | 55 -> 63 | 55 -> 64 |

`seq03_small` 的 top-1 只增加 `1/64`（+1.56 个百分点），但 usable geometry 增加
9 帧，R@20 增加 8 帧，R@100 增加 9 帧，平均 top-1 RIoU 约从 `0.5682` 增至
`0.5789`。因此 S7 readout/RPN 已经修复了候选覆盖，剩余瓶颈转移到 ROI 排序和
native/S7 分路仲裁，不能再把下一步定义为继续扩大 S7 RPN。

逐帧结果显示：S7 分路 top candidate 在 `seq03_small` 有 7 帧成为最终 top-1，7 帧
全部正确，其中只有 `real_seq03[176]` 是新的 top-1 gain；另有 8 帧的 S7 分路第一
候选本身正确，却被错误 native 候选压在后面。`seq02_far` 和 `seq02_dark` 中 S7
没有抢占任何 top-1，原有能力与 MCML 完全保持。

这证明当前全局两标量 affine 校准无法表达“同一帧内 native 与 supplement 的条件
竞争”。另外 `seq03_small` 的 raw detection 数量由 `15697` 增至 `29890`，后续
source gate 通过后还需做独立的延迟/显存检查。

### 8.8 已执行阶段：source-only lane arbitration v1

新增训练模式：

```text
--train-components s7_lane_arbitration
```

它加载并冻结 epoch 1 的完整 S7 merge checkpoint，只训练一个零初始化、输出幅度
受限的 `S7LaneArbitrator`。仲裁器只对 supplement S7 的 pre-NMS logit 加残差，输入
包括 ROI embedding、S7 原始 logit、native top logit；native 分路分数、DINO、native
S14 RPN、S7 readout/RPN、ROI 回归和全局 affine 均不更新。训练损失仍然只由 source
GT 无梯度挖掘 retention/gain pair，NMS 和 Recall 不进入损失。

配置文件：

```text
crane_project/configs/
  crane_symeood_scoped_dino_lowlight_s7_lane_arbitration_v1.py
```

服务器 source-only 训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_rotated_labeller.py \
  --data-root crane_project/data/crane_grab/ \
  --source-train-datasets train:train train_sim:train \
  --source-val-datasets val:val \
  --dinov2-repo third_party/dinov2 \
  --dinov2-checkpoint pretrained/dinov2_vitl14_pretrain.pth \
  --dinov2-model dinov2_vitl14 \
  --dino-gpus 1 2 \
  --head-gpu 0 \
  --legacy-sdpa-query-chunk 512 \
  --dino-height 600 \
  --dino-max-long-side 1333 \
  --patch-size 14 \
  --rpn-feat-channels 256 \
  --roi-fc-channels 1024 \
  --roi-samples 256 \
  --proposal-count 2000 \
  --max-detections 2000 \
  --roi-nms-iou-thr 0.5 \
  --s7-residual \
  --s7-channels 128 \
  --s7-rpn-feat-channels 128 \
  --s7-proposal-count 500 \
  --s7-nms-pre 2000 \
  --s7-anchor-sizes 16 32 64 128 256 \
  --s7-merge-init-bias -2.0 \
  --s7-lane-hidden 32 \
  --s7-lane-max-adjustment 2.0 \
  --s7-lane-base-epoch 1 \
  --train-components s7_lane_arbitration \
  --source-small-repeat 1 \
  --epochs 4 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --max-grad-norm 10 \
  --warmup-iters 1000 \
  --warmup-ratio 0.001 \
  --lr-steps 2 3 \
  --selection-epochs 1 2 3 4 \
  --checkpoint-interval 1 \
  --init-checkpoint \
    work_dirs/dino_teacher_s7_retention_merge_v1/labeller_epoch_01_source_only.pth \
  --feature-cache-dir \
    work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache \
  --work-dir work_dirs/dino_teacher_s7_lane_arbitration_v1 \
  --skip-target-eval \
  --seed 0 \
  --out-json work_dirs/dino_teacher_s7_lane_arbitration_v1/train_result.json
```

这条命令只加载固定 epoch 1 完整 checkpoint，不接受 epoch 2/3 作为仲裁基线；target
不会被读取。只有 `source_exact_retention`、source full/small top-1 和 MCML 全部通过
后，才允许对新 checkpoint 做一次固定三切片 target-dev 对照。

新 checkpoint 必须先通过原有 `677/677` exact-retention source gate；没有通过时，
保留 epoch 0 native 安全权重。source gate 通过后最多做一次固定三切片 target-dev
对照，不使用 target 调参，也不直接做完整 test 或 final validation。

### 8.9 2026-08-01 lane arbitration v1 结果

上述 v1 已完成 source-only 训练，结果文件对应 `protocol_version=12`。DINO、native/S7
RPN、ROI head 和全局 affine 全部冻结，只训练 33,057 个 lane 参数；target 未读取。

| epoch | full top1 / MCML | small top1 / MCML | retained / lost / gained | S7 top1 | gate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 677 / 3 | 303 / 3 | 677 / 0 / 0 | 0 | PASS，正式回退 |
| 1 | 690 / 3 | 316 / 3 | 672 / 5 / 18 | 107 | FAIL |
| 2 | 688 / 3 | 313 / 3 | 673 / 4 / 15 | 79 | FAIL |
| 3 | 688 / 3 | 315 / 3 | 670 / 7 / 18 | 105 | FAIL |
| 4 | 687 / 3 | 315 / 3 | 669 / 8 / 18 | 108 | FAIL |

v1 的总体指标比全局 affine 更高，但所有训练 epoch 都破坏旧正确帧。四个 source 帧在
全部 epoch 持续丢失：`real_seq07/215`、`real_seq07/220`、`sim_seq10/116`、
`sim_seq10/20`。因此正式 `best_epoch=0`，selected checkpoint 是回退行为，不能拿
epoch 1 当部署模型，也没有授权 target 或完整 test。

### 8.10 2026-08-01 lane arbitration v2 + gain replay 结果

v2 使用动态 current-adjusted top-4 hard negatives，并将 7 个 source gain 帧重复 8 次，
形成 49 条额外 replay 记录；结果文件必须以 `protocol_version=13` 且
`source_selected_checkpoint` 指向 `dino_teacher_s7_lane_arbitration_v2` 为准。此前同名
副本中若只把该路径写成 `_v1`，其数值内容相同，但元数据路径错误，不作为权威副本。

| epoch | full top1 / MCML | small top1 / MCML | retained / lost / gained | S7 top1 | val lane mean / max | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 677 / 3 | 303 / 3 | 677 / 0 / 0 | 0 | 0 / 0 | PASS，正式回退 |
| 1 | 681 / 3 | 314 / 3 | 647 / 30 / 34 | 400 | 1.9939 / 2.0 | FAIL |
| 2 | 683 / 3 | 315 / 3 | 650 / 27 / 33 | 391 | 1.9882 / 2.0 | FAIL |
| 3 | 683 / 3 | 315 / 3 | 650 / 27 / 33 | 390 | 1.9816 / 2.0 | FAIL |
| 4 | 683 / 3 | 315 / 3 | 650 / 27 / 33 | 390 | 1.9806 / 2.0 | FAIL |

v2 没有学到条件化仲裁，而是接近对每帧全部 S7 候选统一加最大允许值 `+2`。训练侧
mean adjustment 从 epoch 1 的 `0.0586` 上升到 epoch 3/4 的约 `1.997`；source-val
上约 390--400/738 帧由 S7 成为 top-1。prior loss 到 epoch 4 已增至约 `0.03987`，
仍无法阻止 tanh 输出饱和。gain replay 放大了极少数正向样本，却把错误 S7 候选一起
抬高，结果明显劣于 v1。

该轮 isolation 是干净的：DINO 和所有冻结 head 参数未改变，target 没有用于训练、
选择或评估。失败属于目标函数/参数化问题，不是数据泄露或训练崩溃。正式
`best_epoch=0`，不进行 target 或完整 test。

### 8.11 当前唯一下一步：non-positive S7 quality suppression

停止所有正向 lane promotion 和 gain replay。固定旧的 retention-aware affine epoch 1
作为实验起点（`688/738`、`311/350`、lost=1、gained=12），只在 source train 学习
S7 lane 的可用性/质量，并施加同一帧共享的非正值惩罚：

实现采用 source-only 风险预测器。固定 affine 后选择 S7 lane top candidate，并根据其
ROI embedding、raw/affine log-odds、native top log-odds 和 lane gap 预测风险：

```text
r = source-only S7 lane risk logit
strength = clamp(ReLU(r), 0, 1)
delta = -D * strength，D=2，delta in [-2, 0]
z_s7_new = z_s7_affine + delta
```

输出层零初始化，因此训练开始时 `delta=0`，精确复现 affine epoch 1；风险 BCE 在
边界仍有梯度。错误 S7 只有在 native top 正确且距抢占 margin 小于 `0.5` 时才成为
risk pair；S7 top 本身 `RIoU>=0.5` 时只约束 `delta` 保持接近 0，不增加任何正值。

惩罚对该帧所有 S7 logits 一致，保持 S7 内部排序；由于 `delta <= 0`，它不能相对
旧 affine 基线制造新的 S7 overtakes，只能压回低质量 S7 lane。训练标签来自 source
GT（例如 S7 top candidate 是否 `RIoU >= 0.5`）；DINO、native 路径、S7 readout/RPN、
ROI head 和全局 affine 全部冻结。不要再加 gain loss、gain replay、target-derived
阈值或人工切片路由。

正式门槛：

```text
正式 gate：lost = 0，full >= 688，small >= 311，full/small MCML <= 3
```

不再保留 `lost<=1` 的宽松诊断 gate，因为 affine epoch 1 本身已经满足该条件，近似
no-op 的 suppression 不应再次获得 target-dev 资格。只有正式 gate 通过才产生
source-safe accuracy candidate，并最多授权一次固定三切片对照。若该单调抑制仍不能
消除唯一 lost 且至少保留 11/12 个 gain，则关闭 learned S7 merge，正式模型
继续使用 native alpha=0.5。当前 candidate coverage 已达到 full `738/738`、small
`350/350`，因此不先升级完整 LiDeRe readout，也不再扩大 S7 RPN。

### 8.12 2026-08-01 quality suppression 实现与训练裁决

服务器训练已完成，但训练支持度为零，正式裁决仍为 epoch 0：

```text
训练模式：s7_quality_suppression
固定起点：dino_teacher_s7_retention_merge_v1/labeller_epoch_01_source_only.pth
唯一可训练模块：s7_quality_suppressor.*
输出：每帧一个共享 delta in [-2, 0]
正向 promotion：禁止
gain replay：禁止
target：未读取
协议版本：14（原训练结果）
source train risk pairs：0/2781
四个 epoch 的 delta：全部为 0
affine 候选结果：full 688/738，small 311/350，retained 676/677
official best_epoch：0，S7 disabled
```

配置：

```text
crane_project/configs/
  crane_symeood_scoped_dino_lowlight_s7_quality_suppression_v1.py
```

训练程序会强制 `--skip-target-eval`、`full>=688`、`small>=311` 和 exact retention。
训练起点虽然是 affine epoch 1，但如果全部 epoch 失败，正式 best checkpoint 仍回退到
S7-disabled native alpha=0.5，而不是回退到不安全的 affine checkpoint。

结果分析确认：2781 个 source-train 帧中有 2755 个 preserve pair、26 个 S7 top-1
错误帧，但没有一帧同时满足 `native top correct + S7 top wrong + within margin`。
因此 risk/retention loss 始终为 0，只有 preserve BCE 下降；这不是有效 suppression 学习，
而是风险标签无正样本造成的 no-op。

2026-08-02 将协议升级为 v15，增加训练前 source-only support audit。预检复用正式训练的
同一前向和同一风险判定，并报告全部 S7 错误帧的 native/S7 RIoU、affine gap、margin
排除原因和序列分布。若 `risk_pair_count < 1`，程序在 epoch 1 前停止，保留 epoch 0，
写出 `SOURCE_ONLY_TRAINING_SKIPPED_ZERO_S7_QUALITY_RISK_SUPPORT`，不再产生伪收敛训练。

## 9. 当前研究状态

已经完成：

- 小目标尺度与 RPN 覆盖归因；
- RPN-to-ROI 逐级 attrition；
- source-only ROI classifier、Pairwise V1/V2；
- source-safe `alpha=0.5` 分类器权重插值；
- source-selected DINO ROI NMS `0.5`；
- `[7,14,28]` 插值式多尺度只读上界；
- LiDeRe-inspired S7 readout/RPN 第一版 source-only 训练和严格回退；
- epoch 1 的 1 个 lost / 12 个 gained source 帧冲突审计；
- retention-aware 全局 affine merge 的 source 训练和一次固定三切片 target-dev 诊断；
- lane arbitration v1 训练、exact-retention 拒绝和正式 epoch 0 回退；
- dynamic top-4 + gain replay 的 lane arbitration v2 训练、饱和诊断和正式 epoch 0 回退；
- source-only lane-wide non-positive S7 quality suppression 训练、零风险支持诊断、正式
  epoch 0 回退，以及 v15 预检/早停保护；
- 统一因果时序第一版、候选级 continuous/relative quality、source attribution 和递归
  immediate-override 审计；
- 阶段三 source-only candidate student 训练、source gate 和一次固定三切片 target-dev；
- 最后一次 source-only 静态域泛化 ranker 的 4 个 epoch 训练与 exact-retention 回退；
- GTX 1080 三卡延迟与参数量基线。

尚未完成：

- `seq03_small` 所代表的未知小目标最终 Top-1 排序改进；
- 一次结构不同、source-only、native-protected 的通用小目标选择实验；
- 从原始图像、DINO、S14/S7、RPN/ROI、候选选择到最终 OBB 的统一训练/推理入口；
- 不依赖历史 `train_result.json`、固定 work-dir 或审计脚本的最终可部署 checkpoint；
- DINO teacher 与可训练轻量 student 的蒸馏/联合训练闭环；
- 满足预注册 FPS/latency 目标的实时推理；
- 不依赖预定义困难片段、真正未知小目标序列上的独立 final validation；
- 模型完全冻结后的一次性完整 test。

当前正式裁决：

```text
保持暗光 DINO 和 source-safe alpha=0.5 ROI 分类头作为唯一正式基线；
当前最佳 source-safe S7 实验候选更新为 high-resolution ROI ranker epoch 3：
687/738、small 310/350、lost=0；它只因 absolute gate 各差 1 而仍为诊断候选；
S7 已把固定 seq03_small 的 R@100 候选覆盖提高到 64/64，但最终 Top-1 仍为 50/64；
阶段三 student 已通过 source gate，但固定 target-dev 未达到预注册 51/64；
最后一次静态域泛化 ranker raw 最优为 695/738、318/350，但 lost=1，正式 best_epoch=0；
关闭继续增加当前 static ranker epoch、重复 target-dev 和 target-derived 路由；
旧 static/selective ranker 分支探索结束，但 high-resolution ROI ranker 已形成新的安全候选；
下一阶段只做冻结 epoch 3、共享前向的 source-only promotion-margin 审计；
通过后才做一次固定 target-dev，再统一训练/推理入口和 checkpoint；
之后才进行轻量 student、实时化、真正未知序列泛化与完整 final test；
EKF/卡尔曼仍留到大论文后半部分。
```

## 10. 新对话快速接手摘要

### 10.1 当前可用基线

```text
DINO head checkpoint:
work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth

source full:  677/738，MCML=3
source small: 303/350，MCML=3
old source-correct frames lost: 0
DINO ROI NMS IoU: 0.5
```

暗光大目标分支的已验证结论仍保留：`seq02_dark` 原始 DINO 约 `29/33`，因果稳定化
约 `32/33`、MCML=1。小目标修改不得破坏该能力。

### 10.2 当前小目标瓶颈

```text
seq02_far: RPN geometry 已基本存在；当前 highres 固定诊断为 38/40
seq02_dark: 当前 highres 固定诊断为 29/33，作为不回退保护片段
seq03_small: short side ~= 1.124 DINO token；S14 RPN@2000 = 45/64
highres S7: target R@100 = 55/64 -> 64/64，但 Top-1 仍为 50/64，MCML=6
```

空间采样曾限制候选覆盖，但已经被 highres S7 的 R@100 结果实证缓解；当前根本瓶颈是：既有
source split 中不存在可同时覆盖 real/sim、跨 sequence 且保留独立 real hard holdout 的
native-wrong/S7-correct 监督。因此不能从现有 source 数据训练并证明一个 native-safe 的跨域
统一排序器。`seq03_small` 是 target-dev 症状，不是训练或阈值调节对象。

### 10.3 初始 S7 与 merge 阶段结论

```text
LiDeRe-inspired S7 readout learned useful extra candidates
best raw source epoch = 2: full 688/738, small 318/350
exact retention = 654/677, FAIL
official selected best_epoch = 0, S7 disabled
target diagnosis was not run
```

这表示 readout 方向有正信号，但当前简单的 `native proposals + S7 proposals -> shared ROI/NMS` 会让新增错误候选抢走原正确候选。不能把 `best_epoch=0` 当成 S7 改进结果，
也不需要重复运行同一训练命令或 target/full-test。

随后完成的全局 affine merge 和固定 target-dev 诊断证明：

```text
source affine epoch 1: full 688/738, small 311/350, retained 676/677
seq02_far:   38/40 -> 38/40
seq02_dark:  29/33 -> 29/33
seq03_small: 50/64 -> 51/64, R@100/geometry 55/64 -> 64/64
```

候选覆盖已解决，但 exact retention 仍差 1 帧。lane arbitration v1 最少 lost=4、最高
full=690/small=316；v2 最少 lost=27，并饱和到约 `+2`。两轮正式 `best_epoch=0`，
没有可供 target 测试或部署的新 checkpoint。

### 10.4 历史停止裁决：quality suppression 阶段

> 本裁决只关闭当时的 lane/suppression 分支；后续第 10.5--10.17 节已经完成新的时序、
> relative-quality、阶段三和 static-ranker 实验。当前路线以第 10.17 节为准。

```text
quality suppression v1：zero risk support / no-op / best_epoch=0
关闭当前 learned S7 merge
保留 native S14 alpha=0.5 正式基线
不运行 target-dev、完整 test，也不增加 epoch 或原样重训
```

v15 预检只用于固化失败证据和防止误训，不构成新的模型实验。若未来提出新的风险标签
v2，必须先基于这次 source-only audit 单独预注册，并继续禁止 target-derived 阈值；在此
之前不开放 target 调参、完整 test、压缩、完整 LiDeRe 升级或实时性优化。

### 10.5 保留方案：统一跨尺度因果时序候选选择

2026-08-02 将后续统一方向固定为：不再按 `seq02_dark` / `seq03_small` 或任何预定义
target 片段路由，而是在所有连续视频上统一运行跨尺度候选选择。现有证据表明两类困难
都包含“正确候选存在但逐帧 top-1 不可靠”，但来源不同：暗光大目标主要是语义排序
坍塌，S14 DINO 已将其恢复到 `29/33, MCML=1`；小目标首先受 S14 空间分辨率限制，
S7 将 `seq03_small` 的 R@100/usable geometry 从 `55/64` 提升到 `64/64` 后，剩余
问题才转移到排序。

保留的统一结构为：

```text
每帧 native S14 candidates + frozen S7 candidates
  -> source-supervised candidate-level RIoU/quality prediction
  -> source-selected causal temporal candidate association/reranking
  -> native-safe top-1 selection with reset/fallback
  -> 已有因果框稳定器仅平滑最终框的 w/h/periodic-theta
```

候选级质量头对每个 native/S7 ROI 独立预测质量，监督使用 source GT 的连续最大 RIoU，
不再使用整帧 lane-wide `delta` 或稀疏 risk pair。因果关联至少联合 rotated IoU、归一化
中心位移、对数尺度变化、周期角度变化和 DINO ROI 外观相似度；小目标不能只依赖 IoU。
S7 单帧高分不得直接抢占 native，必须获得连续多帧、跨尺度一致或高质量/高外观一致
证据。关联失败、序列切换、帧号不连续或长时间低置信时清空状态并回退 native top-1。

该方案与现有因果框稳定器职责不同：现有稳定器位于 top-1 选择之后，不改变候选分数、
顺序或输出存在性；新模块位于 top-1 选择之前，解决已有正确候选的时序排序。早期
BrightAug 手工时序重锚失败时，暗光正确候选位于约 top-5708；当前 S7 已将小目标正确
候选推进到 R@20 `63/64`、R@100 `64/64`，因此不能把早期失败直接外推到当前候选池，
但必须采用 source-only 预检和固定停止门，避免重新扫描 target 权重。

协议边界：

```text
禁止：sequence id、预定义 dark/small scope、target GT、target-derived threshold
训练/选择：source train + source validation continuous sequences only
安全门：source old-correct lost=0；full>=688；small>=311；MCML<=3
时序门：source sequence MCML/DFR/ACI 不回退；reset 后精确回到 native fallback
target：所有参数冻结后只做固定诊断；最终泛化必须使用未参与设计的新序列
```

实施前必须统一 MCML 口径。当前固定 `seq03_small` RIoU top-1 诊断记录为 `MCML=6`，
另有完整组合/系统 evaluator 口径报告 `MCML=14`；后续结果必须同时明确 RIoU 阈值、
score threshold、silence 处理、候选过滤和序列 reset 规则，不能混用。

对应的 2025--2026 文献检索、原始来源和证据--论点映射已单独保存在
`docs/20260802_unified_temporal_candidate_literature_2025_2026.md`。当前实施优先级据此
收紧为：先固定 native+S7 候选池，以归一化中心距离、rotated IoU 和 DINO ROI 外观
相似度为核心做轻量 source-only 因果关联，再加入尺度/周期角度一致性和 native-safe
fallback。PQA/RARE 只作为“分类分数不等于定位质量、候选应相对排序”的文献依据；
BrightAug PQA v1--v3 已关闭，不能据此启动 PQA v4 或 quality-only top-1。完整 LiDeRe、
可训练时序 adapter 和 target-train 自监督均后置，只有第一版 source 预检显示明确缺口时
才重新授权。

### 10.6 统一方案第一阶段代码状态（2026-08-03）

第一版实现不是新的 candidate quality head，而是在固定
affine epoch 1 的 native/S7 post-NMS top-100 候选上，只拟合六个非负多线索权重：

```text
calibrated score logit
normalized center distance
rotated IoU
log-scale consistency
pi-periodic angle consistency
DINO ROI appearance cosine similarity
```

训练记录按 `split/seq/frame` 排序，只在帧号连续时使用上一帧 source GT usable candidate
构造关联监督；推理只使用上一帧已选择候选。首帧、序列切换、帧号缺口或连续性失败均
精确回退 native top-1。非 native 候选默认需要连续两帧确认后才能接管。source-small
指标从完整连续 source-val pass 中切片统计，不再对 small 子集单独运行并错误跨缺帧关联。

新增入口：

```text
crane_project/utils/s7_temporal_association.py
crane_project/configs/
  crane_symeood_scoped_dino_lowlight_s7_temporal_association_v1.py
--train-components s7_temporal_association
--s7-temporal-association
```

部署配置显式使用 `scope_policy=all_frames`、`scope_manifest=None`，不读取 dark/small scope
或 sequence id 作为模型特征。现有 post-selection OBB stabilizer 保持独立，仍只平滑
最终框的尺度和周期角度。

source-only 第一轮命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_rotated_labeller.py \
  --data-root crane_project/data/crane_grab/ \
  --source-train-datasets train:train train_sim:train \
  --source-val-datasets val:val \
  --dinov2-repo third_party/dinov2 \
  --dinov2-checkpoint pretrained/dinov2_vitl14_pretrain.pth \
  --dinov2-model dinov2_vitl14 \
  --dino-gpus 1 2 \
  --head-gpu 0 \
  --legacy-sdpa-query-chunk 512 \
  --dino-height 600 \
  --dino-max-long-side 1333 \
  --patch-size 14 \
  --rpn-feat-channels 256 \
  --roi-fc-channels 1024 \
  --roi-samples 256 \
  --proposal-count 2000 \
  --max-detections 2000 \
  --roi-nms-iou-thr 0.5 \
  --s7-residual \
  --s7-channels 128 \
  --s7-rpn-feat-channels 128 \
  --s7-proposal-count 500 \
  --s7-nms-pre 2000 \
  --s7-anchor-sizes 16 32 64 128 256 \
  --s7-merge-init-bias -2.0 \
  --s7-temporal-association \
  --s7-temporal-base-epoch 1 \
  --s7-temporal-margin 0.5 \
  --s7-temporal-retention-weight 2.0 \
  --s7-temporal-gain-weight 1.0 \
  --s7-temporal-prior-weight 0.01 \
  --s7-temporal-max-candidates 100 \
  --s7-temporal-min-confirmations 2 \
  --s7-temporal-override-margin 0.25 \
  --s7-temporal-max-center-distance 3.0 \
  --s7-temporal-min-riou 0.05 \
  --s7-temporal-min-appearance 0.20 \
  --s7-source-min-full-top1 688 \
  --s7-source-min-small-top1 311 \
  --s7-source-max-mcml 3 \
  --train-components s7_temporal_association \
  --source-small-repeat 1 \
  --source-retain-max-top1-drop 0 \
  --epochs 4 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --max-grad-norm 10 \
  --warmup-iters 1000 \
  --warmup-ratio 0.001 \
  --lr-steps 2 3 \
  --selection-epochs 1 2 3 4 \
  --checkpoint-interval 1 \
  --init-checkpoint \
    work_dirs/dino_teacher_s7_retention_merge_v1/labeller_epoch_01_source_only.pth \
  --feature-cache-dir \
    work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache \
  --work-dir work_dirs/dino_teacher_s7_temporal_association_v1 \
  --skip-target-eval \
  --seed 0 \
  --out-json \
    work_dirs/dino_teacher_s7_temporal_association_v1/train_result.json
```

第一轮仍执行正式 gate：`lost=0`、full `>=688/738`、small `>=311/350`、full/small
`MCML<=3`，并要求完整 source 连续流的 DFR 不高于 native、ACI 不低于 native（沿用
项目 evaluator 的 `35°` ACI 上限）。如果所有 epoch 失败，
`labeller_best_source_only.pth` 继续是 S7-disabled epoch 0；不得拿原始 epoch checkpoint
做 target 调参。只有 `best_epoch>0` 才授权一次冻结三切片诊断。

### 10.7 2026-08-03 时序第一版 source 结果与安全修补

服务器已完成第一版 source-only 训练，权威结果为外部迁移文件
`/Users/mac/Downloads/Copy of train_result.json`，协议版本 `16`。四个 epoch 的离散
输出完全相同：full `680/738`、small `305/350`、MCML 均为 `3`，exact retention
`677/677`，`lost=0`，`gained=3`。因此时序关联消除了 affine merge 的唯一 source 丢失，
但没有达到正式 `688/311` 门槛，正式 `best_epoch=0`，target 未读取，仍保留 native
S14 alpha=0.5 正式模型。时序指标从 DFR `0.08464` 降至 `0.08272`，ACI 从 `0.82687`
升至 `0.83092`；候选 R@100 达到 full `738/738`、small `350/350`，说明当前主要短板
是选择增益而不是候选覆盖。

结果显示 source train 中只有 `7/2781` 个 gain pair，无法为六个全局 cue 权重提供足够
稠密的增益监督。该轮证明 native-safe 时序选择可实现 `lost=0`，但不能取代正式基线。

随后增加了三项不改变模型目标的安全诊断：

```text
1. temporal_association.source_selected=True 时，运行时强制检查 checkpoint best_epoch、
   source gate、exact retention 和 full/small/MCML；失败即拒绝启用 S7，防止 epoch-0
   fallback checkpoint 被时序配置误部署。
2. 训练前记录初始六权重的 source temporal summary，区分初始化收益与训练收益。
3. 每个 source temporal summary 记录 candidate margin、continuity、override、pending
   confirmation、reset 等阻断原因；不使用 target、不改变 checkpoint 选择。
```

相关代码入口：`source_selected_checkpoint_gate`、
`summarize_temporal_association_audit`、`CausalTemporalCandidateSelector` 的诊断字段。
本地回归结果：相关测试 `114 passed`，py_compile 与 `git diff --check` 均通过。候选级
continuous-RIoU quality head 随后作为唯一 source-only 后续阶段实现，尚未训练。

### 10.8 2026-08-03 候选级连续 RIoU quality head 实现状态

由于 source train 的 gain pair 稀疏来自视频分布差异，本阶段不再依赖 gain-pair miner、
gain replay 或正向 lane promotion。新增 `S7CandidateQualityHead`，对固定 affine
epoch-1 的 native/S7 post-NMS top-100 候选逐候选计算 source GT max-RIoU，并使用
`weighted_smooth_l1(sigmoid(logit), target)`（权重 `1+3*target`）进行 dense supervision。
quality logit 只作为第七个 causal temporal cue；六个原始 cue 权重、DINO、native/S7
proposal、ROI head 和 affine calibrator 全部冻结。quality head 输出层零初始化，因此
训练前的候选排序与已审计时序基线完全一致。

部署和训练均保持统一全序列策略：`scope_policy=all_frames`、`scope_manifest=None`，
native fallback、两帧确认、sequence/frame-gap reset、DFR/ACI 与 exact-retention gate
不变；target 不参与训练、checkpoint 选择或运行时路由。新增入口为：

```text
crane_project/utils/s7_temporal_association.py
crane_project/configs/crane_symeood_scoped_dino_lowlight_s7_temporal_quality_association_v1.py
--s7-temporal-quality-head
```

source-only 运行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_rotated_labeller.py \
  --data-root crane_project/data/crane_grab/ \
  --source-train-datasets train:train train_sim:train \
  --source-val-datasets val:val \
  --dinov2-repo third_party/dinov2 \
  --dinov2-checkpoint pretrained/dinov2_vitl14_pretrain.pth \
  --dinov2-model dinov2_vitl14 \
  --dino-gpus 1 2 --head-gpu 0 \
  --legacy-sdpa-query-chunk 512 --dino-height 600 \
  --dino-max-long-side 1333 --patch-size 14 \
  --rpn-feat-channels 256 --roi-fc-channels 1024 \
  --roi-samples 256 --proposal-count 2000 --max-detections 2000 \
  --roi-nms-iou-thr 0.5 \
  --s7-residual --s7-channels 128 --s7-rpn-feat-channels 128 \
  --s7-proposal-count 500 --s7-nms-pre 2000 \
  --s7-anchor-sizes 16 32 64 128 256 --s7-merge-init-bias -2.0 \
  --s7-temporal-association --s7-temporal-quality-head \
  --s7-temporal-quality-hidden 128 --s7-temporal-quality-loss-weight 1.0 \
  --s7-temporal-base-epoch 1 --s7-temporal-max-candidates 100 \
  --s7-temporal-min-confirmations 2 --s7-temporal-override-margin 0.25 \
  --s7-temporal-max-center-distance 3.0 --s7-temporal-min-riou 0.05 \
  --s7-temporal-min-appearance 0.20 \
  --s7-source-min-full-top1 688 --s7-source-min-small-top1 311 \
  --s7-source-max-mcml 3 --train-components s7_temporal_association \
  --source-small-repeat 1 --source-retain-max-top1-drop 0 \
  --epochs 4 --lr 0.001 --momentum 0.9 --weight-decay 0.0001 \
  --max-grad-norm 10 --warmup-iters 1000 --warmup-ratio 0.001 \
  --lr-steps 2 3 --selection-epochs 1 2 3 4 --checkpoint-interval 1 \
  --init-checkpoint \
    work_dirs/dino_teacher_s7_retention_merge_v1/labeller_epoch_01_source_only.pth \
  --feature-cache-dir \
    work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache \
  --work-dir work_dirs/dino_teacher_s7_temporal_quality_association_v1 \
  --skip-target-eval --seed 0 \
  --out-json \
    work_dirs/dino_teacher_s7_temporal_quality_association_v1/train_result.json
```

本阶段仍只接受正式 gate：`lost=0`、full `>=688/738`、small `>=311/350`、full/small
`MCML<=3`，以及 DFR/ACI 不劣化。若所有 epoch 不通过，best checkpoint 必须回退到
native S14 epoch 0；未通过 gate 前不得读取 target 或运行完整 test。

### 10.9 2026-08-03 固定 epoch-4 source-only 时序归因审计

quality temporal 的四个 epoch 均为 full `681/738`、small `306/350`、`lost=0`，正式
`best_epoch=0`。为判断差距来自第七个 quality cue、七 cue 融合、margin/continuity，
还是两帧 confirmation，新增协议 19 的只读审计模式：

```text
--source-temporal-attribution-audit
--source-temporal-attribution-epoch 4
--eval-only-checkpoint <labeller_epoch_04_source_only.pth>
--skip-target-eval
```

该模式只对固定 rejected epoch-4 运行一次 source-val 前向，不训练、不更新参数、不重新
选择 best epoch、不读取 target。每帧在不改变 selector 输出和时序状态的前提下记录：

```text
native fallback
quality-only 排序及正确候选 rank
七 cue fused argmax
margin 反事实
margin + continuity 的 pre-confirmation 反事实
pending confirmation 候选
最终 selected candidate
```

输出 `source_temporal_attribution_audit` 会分别汇总 full 和 source-small 的 top-1、相对
fallback 的 gain/loss、quality rank、pending correctness，并计算当前 final 距正式门槛的
`+7 full/+5 small` 缺口。只有 pre-confirmation 同时达到 full `>=688`、small `>=311`
且相对 fallback `lost=0`，才输出
`ALLOW_ONE_BOUNDED_CONFIRMATION_RULE_REVISION`；否则输出
`CLOSE_CURRENT_QUALITY_TEMPORAL_MERGE_KEEP_NATIVE_BASELINE`。这里的 pre-confirmation 是
固定当前时序状态的一步反事实，不会递归更新后续时序状态，因此即使通过也只授权一次
受限 confirmation 规则修改，仍需重新通过完整 source gate。

实现入口为 `temporal_selection_attribution`、
`summarize_temporal_readonly_attribution` 和
`build_source_temporal_attribution_audit`。本地检查通过：相关测试 `123 passed`、
`py_compile` 与 `git diff --check` 通过。

### 10.10 2026-08-04 source-only recursive immediate-override 实验

考虑到后半部分将增加自适应协方差 EKF/卡尔曼后处理，时序候选不再把“两帧确认”视为
最终不可放宽的约束。但仍保留 native fallback、candidate top-100、margin、continuity、
sequence/frame-gap reset 和 source-only 隔离；不允许 quality-only 排序或无条件 S7
promotion。

新增显式只读入口：

```text
--source-temporal-immediate-override-audit
```

该入口固定读取 rejected epoch-4 checkpoint，并仅在
`candidate_margin_ok && candidate_continuity_ok` 时将运行时 confirmation 设为 1 帧。
它会真实递归更新下一帧的 temporal state，用于验证此前一步反事实 `690/738、312/350`
是否能够在真实顺序推理中保持。该模式不训练、不更新参数、不选择 checkpoint、不读取
target，输出写入 `source_temporal_immediate_override_audit`。

source-only 运行后仍需记录 full/small top-1、MCML、lost、DFR 和 ACI；这些结果是第一部分
候选选择的实验结果，后续再由自适应协方差 EKF/卡尔曼模块处理短时抖动和观测不确定性，
不能把 immediate-override 结果直接当作最终部署模型。

### 10.11 2026-08-04 阶段二B结果与固定三切片诊断入口

阶段二B relative-quality source-only 训练已完成。权威迁移结果为
`/Users/mac/Downloads/Copy of Copy of Copy of Copy of train_result.json`，正式选择
`source.best_epoch=4`：full `691/738`、small `312/350`、full/small MCML 均为 `3`，
exact retention `677/677`、`lost=0`，R@100 达到 `738/738`、`350/350`；DFR 和 ACI
也通过 source 时序非退化门。训练只更新 candidate quality head，target 未读取。相对
pointwise immediate-override 初始化，离散 top-1 没有继续增加，因此不再增加 epoch 或
扫描 relative loss 超参数；本轮结论是“source-safe 排序细化通过”，不是把 relative loss
单独宣称为新的 top-1 增益来源。

新增严格只读入口：

```text
crane_project/tools/dino_teacher_s7_temporal_fixed_target_audit.py
```

该入口不同于历史 relaxed-gate 工具：只接受当前 source-selected epoch-4
relative-quality checkpoint，要求 `lost=0`、full `>=688`、small `>=311`、MCML `<=3`、
DFR/ACI 非退化，并验证 checkpoint 内部 source gate、relative-quality、quality head、
`min_confirmations=1` 和 S7 inference 元数据。候选 checkpoint 必须与训练结果中的
`source_selected_checkpoint` 路径一致。target 切片、阈值和门槛全部在代码内固定，命令行
不能修改：

```text
seq02_far:   test/real_seq02[2..41]，baseline 38/40，MCML=1
seq02_dark:  test/real_seq02[137..169]，baseline 29/33，MCML=1
seq03_small: test/real_seq03[129..192]，baseline 50/64，MCML=6，R@100=55/64
```

阶段三授权门为：far/dark top-1 与 MCML 不回退并达到上述绝对参考线；seq03_small
至少 `51/64`、MCML `<=6`、R@100 `64/64`。DFR/ACI 继续完整报告，但考虑后续独立
EKF/卡尔曼阶段，本次不作为 target-dev 硬门。运行过程 optimizer steps 为 0，参数版本
必须保持不变；输出只决定是否授权阶段三学生训练，不构成部署或完整 test 证据。

### 10.12 2026-08-04 source-only attribution 固化入口

固定三切片 target-dev 的实际结果为：`seq02_far=39/40`、`seq02_dark=29/33`，
`seq03_small=50/64`；后者的 R@100 已达到 `64/64`，但没有达到预注册的严格
Top-1 `51/64`，因此阶段三仍未授权。该差距不能事后放宽为通过。

为分析“正确候选已经进入前 100，但没有排到第一”的来源，新增严格只读入口：

```text
crane_project/tools/dino_teacher_s7_temporal_source_attribution_audit.py
```

入口只接受 source-selected epoch-4 relative-quality checkpoint，内部固定
`train:train + train_sim:train`、`val:val`、DINO ViT-L/14、S14/S7 结构、relative
quality 配置和 `seed=0`；强制 `--eval-only-checkpoint`、`--skip-target-eval` 和
`--source-temporal-attribution-audit`。它会在 source full/short-token small 两组上
记录 native fallback、quality-only、七 cue fused、margin、pre-confirmation 和 final
selection 的逐帧归因，不训练、不更新参数、不读取 target、不选择 checkpoint。

只有当 source-only pre-confirmation 同时达到 full `>=688`、small `>=311` 且相对
fallback `lost=0` 时，才允许把结果解释为一次受限 confirmation-rule 新实验的依据；
否则关闭 learned S7 merge，继续保留 native S14 `alpha=0.5` 正式基线。该 attribution
结果本身也不构成部署或完整 test 证据。

### 10.13 2026-08-05 递归 immediate-override 审计入口

epoch-4 source attribution 已输出
`ALLOW_ONE_BOUNDED_CONFIRMATION_RULE_REVISION`：margin+continuity 的
pre-confirmation 在 full `691/738`、small `312/350` 上相对 native fallback
均 `lost=0`。因此新增严格入口：

```text
crane_project/tools/dino_teacher_s7_temporal_immediate_override_audit.py
```

该入口复用 source-selected epoch-4、relative-quality、source-only 校验，只将审计模式
切换为 `--source-temporal-immediate-override-audit`，用于真实递归更新下一帧时序状态。
它不训练、不更新参数、不读取 target、不选择 checkpoint。该结果用于确认安全选择规则能否
在连续视频中保持，不能把 `seq03_small` 的 `50/64` target-dev 结果事后改判为通过。

若递归审计仍保持 source full `>=688`、small `>=311`、`lost=0`，才进入阶段三学生训练；
阶段三仍须先进行 source-only gate，之后才有资格进行新的固定 target-dev 诊断。当前证据表明
`seq03_small` 的候选覆盖已经解决，14 个 target Top-1 失败不能归因于 confirmation；后续若
仍要提升它，应由阶段三学生训练改善跨域排序，而不是继续重复 quality-only 或无条件 S7 promotion。

### 10.14 2026-08-05 阶段三 source-only 学生训练实现

阶段三已按统一模型路线实现，但尚未运行。由于当前数据只有 `train/train_sim/val/test`，没有
与 test 隔离的无标签 target-train split，本阶段明确禁止读取 test 伪标签；实现的是 source-only
候选排序学生，而不是 target pseudo-label student。

固定 teacher 为阶段二 source-selected epoch 4：DINOv2、native S14 RPN、S7 readout/RPN、
ROI classifier/regressor、global affine calibrator、七 cue scorer 和阶段二 candidate-quality
head 全部冻结。新增独立 `s7_candidate_student_head`，先精确复制 teacher，再仅训练该学生头：

```text
L_stage3 = L_continuous_source_RIoU
         + 0.5 * L_same_frame_relative_quality
         + 1.0 * L_teacher_Bernoulli_distillation
```

短边不超过 4 个 DINO token 的 source GT 帧只在监督损失上乘 `2.0`；该尺度标签不是推理输入，
不读取序列名，也不对 `seq03_small` 或三段 target-dev 做路由。推理仍为同一 native/S7 top-100
候选池、固定七 cue、margin+continuity、one-frame confirmation、native fallback 和 causal reset。

安全约束包括：初始化 checkpoint 必须等于阶段二结果中的 `source_selected_checkpoint`，且通过
`best_epoch=4`、source gate、exact retention；复制后先重新验证阶段二 full/small/DFR/ACI，
不一致即停止。epoch 1--4 只有同时满足 `lost=0`、full `>=688`、small `>=311`、full/small
MCML `<=3`、DFR/ACI 不退化，并且 selection key 严格优于复制 teacher，才可替代 epoch-0
阶段二 fallback。运行入口为：

```text
crane_project/tools/dino_teacher_s7_temporal_student_train.py
crane_project/configs/crane_symeood_scoped_dino_lowlight_s7_temporal_student_v1.py
```

本轮只运行 source-only 阶段三。只有 `best_epoch>0` 才授权下一次固定三切片 target-dev；不得直接
运行完整 test。EKF/卡尔曼仍属于模型排序基本成立后的最终后处理阶段。

### 10.15 2026-08-05 阶段三 fixed target-dev 结果与完整指标边界

阶段三 source-only student 已通过 source gate 后，按预注册协议完成一次固定三切片
target-dev 对照。权威结果文件为
`/Users/mac/Downloads/target_dev_result.json`，候选为
`dino_teacher_s7_temporal_student_v1/labeller_best_source_only.pth`，`best_epoch=4`。
本轮没有训练，`optimizer_steps=0`，DINO、baseline head 和 candidate head 参数均未更新。

target-dev 使用的范围仍是预定义诊断切片，但只用于评估，不用于训练、checkpoint 选择、
参数调节或模型内路由：

```text
seq02_far:   test/real_seq02[2..41]
seq02_dark:  test/real_seq02[137..169]
seq03_small: test/real_seq03[129..192]
```

#### 10.15.1 Top-1、MCML 与候选覆盖

| 切片 | baseline Top-1 | student Top-1 | baseline MCML | student MCML | baseline R@20/R@100 | student R@20/R@100 | target gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `seq02_far` | 38/40 | 39/40 | 1 | 1 | 40/40 | 40/40 | 通过 |
| `seq02_dark` | 29/33 | 29/33 | 1 | 1 | 33/33 | 33/33 | 通过 |
| `seq03_small` | 50/64 | 50/64 | 6 | 6 | 55/55 | 63/64 | 失败 |

`seq03_small` 的 `MCML=6` 确实满足本切片预注册的 `MCML<=6`，因此失败原因不是 MCML，
而是严格 Top-1 门要求 `51/64`，实际仍为 `50/64`。本轮同时确认候选覆盖已经达到
`R@100=64/64`；剩余 14 帧属于候选存在但最终没有排到第一的问题。

#### 10.15.2 target-dev 可用的自定义时序指标

以下指标来自本轮 target-dev evaluator，DFR 为百分比形式，ACI 为无量纲指标：

| 切片 | 模型 | mean Top-1 RIoU | DFR (%/frame) | ACI | temporal override | reset |
|---|---|---:|---:|---:|---:|---:|
| `seq02_far` | baseline | 0.61218 | 6.2213 | 0.91711 | 0 | 0 |
| `seq02_far` | student | 0.61735 | 6.3526 | 0.94417 | 1 | 1 |
| `seq02_dark` | baseline | 0.60913 | 13.2427 | 0.76817 | 0 | 0 |
| `seq02_dark` | student | 0.60913 | 13.2427 | 0.76817 | 0 | 1 |
| `seq03_small` | baseline | 0.56820 | 9.2120 | 0.87517 | 0 | 0 |
| `seq03_small` | student | 0.57436 | 9.4894 | 0.88085 | 4 | 1 |

本轮结果没有导出 `R_center` 和 `TDR_w10`。它们属于完整系统级 evaluator 的指标，不能
由当前 target-dev 的 Top-1 RIoU 摘要可靠反推；后续若进行正式 full-test，必须使用统一的
`R_center` 中心误差阈值、`TDR_w10` 窗口定义、silence 处理和 MCML 口径，并同时输出
`R_center/DFR/TDR_w10/ACI/MCML`，不能只报告 Top-1。

#### 10.15.3 正式裁决与下一步

```text
decision = S7_TEMPORAL_STUDENT_FIXED_TARGET_DEV_FAIL_KEEP_SOURCE_MODEL
eligible_for_deployment = false
eligible_for_final_test = false
```

因此本轮不能授权完整 test。`MCML=6` 在 `seq03_small` 上可以作为诊断结果保留，
但不能抵消预注册的 `51/64` Top-1 条件，也不能事后放宽 gate。

阶段三已经证明 source-safe，但没有在 target-dev 上改善最终小目标排序。后续小论文阶段
不使用自适应协方差 EKF/卡尔曼观测器；下一步只允许做一次 source-only、无时序的静态域泛化
排序实验：固定 S7 候选生成，加入通用亮度/模糊/尺度扰动和候选 hard-negative 相对排序，
不读取 target、不按三段切片路由，并继续要求 `lost=0`。如果该实验仍不能提升
`seq03_small` Top-1，则停止静态排序路线，将当前结果作为“候选覆盖改善、最终排序未完全解决”
的小论文证据，完整 test 留到模型方案最终冻结后一次性执行；EKF/卡尔曼保留给大论文后半部分。

### 10.16 2026-08-06 source-only 静态域泛化 ranker 最终结果与路线关闭

最后一次 source-only 静态域泛化排序实验已经完成。权威迁移结果为
`/Users/mac/Downloads/Copy of train_result.json`，协议版本 `22`，结果裁决为：

```text
decision = SOURCE_ONLY_STATIC_DOMAIN_RANKER_FALLBACK_TARGET_NOT_READ
source.best_epoch = 0
target_dev = null
```

本轮固定 DINOv2、native S14/S7 proposal、ROI head、global affine 和既有时序/quality
组件，只训练 `s7_candidate_static_head`（132609 个参数）。训练监督仅来自
`train + train_sim` 的 source GT，并使用通用 brightness/blur/scale feature-domain
augmentation 和同帧 hard-negative relative ranking；推理不使用时序、不读取序列名、
不按困难切片路由，也不读取 target。

四个 epoch 的 source 结果如下：

| epoch | full Top-1 | small Top-1 | full/small MCML | full R@20/R@100 | small R@20/R@100 | lost/gained | source gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 691/738 | 315/350 | 3/3 | 736/738 | 350/350 | 2/16 | 失败 |
| 2 | **695/738** | **318/350** | 3/3 | 736/738 | 350/350 | **1/19** | 失败 |
| 3 | 693/738 | 315/350 | 3/3 | 736/738 | 350/350 | 1/17 | 失败 |
| 4 | 694/738 | 316/350 | 3/3 | 736/738 | 350/350 | 1/18 | 失败 |

epoch 2 是 raw 指标最好的训练状态，DFR 从 native 的 `0.08464` 降至 `0.07934`，ACI
从 `0.82687` 升至 `0.83428`；但它仍破坏了旧正确帧
`val|real_seq07|215`。其他 epoch 也分别丢失 1--2 个旧正确帧。所有绝对 Top-1、MCML
和候选覆盖条件均已通过，唯一失败项始终是 `exact_old_correct_retention`，因此
`best_epoch=0` 是预注册安全选择器的正确回退，不是 checkpoint 保存或 best-epoch 代码错误。

#### 10.16.1 这是否验证了 seq03_small 可以提升

本实验的研究目标确实是判断“通用静态跨域排序能否在不依赖 target 的前提下，把已经进入
top-100 的小目标正确候选稳定推到 Top-1”，因此它是为 `seq03_small` 提升设计的最后一个
source-only 前置判别实验。但本轮 `target_dev=null`，训练、checkpoint 选择和评估均没有读取
`seq03_small`，所以它不能直接证明 `seq03_small` 已经提升，也不能把 source epoch 2 的
`695/738` 外推成 target 的 `51/64`。

按照预注册顺序，只有 source gate 通过才允许再读取一次固定 target-dev。本轮四个 epoch 均未
通过 exact retention，因而不再运行 `seq03_small` 对照是协议要求，不是漏测。当前对该切片
最后一份有效证据仍为阶段三：Top-1 `50/64`、MCML `6`、R@20 `63/64`、R@100 `64/64`。
这说明候选覆盖已经基本解决，剩余 14 帧主要是跨域最终排序失败；现有静态 ranker 具有明显
source 排序能力，但尚无证据证明其能安全迁移到该 target 切片。

#### 10.16.2 综合停止裁决

```text
formal deployable model:
  native S14 + alpha=0.5 ROI classifier, S7 disabled

stage-3 student:
  source-safe, but fixed target-dev seq03_small remains 50/64
  eligible_for_deployment = false
  eligible_for_final_test = false

static domain ranker:
  raw source gains are diagnostic only
  best_epoch = 0
  target not read
  close current static ranking route
```

至此，当前统一方案的候选生成、时序选择、candidate relative quality、阶段三 student 和
静态域泛化排序实验均已完成。停止增加 epoch、重复 target-dev、放宽事后 gate 或使用
target-derived/manual slice routing。S7 的论文结论应限定为“候选覆盖从 `55/64` 提高到
`64/64`，但 source-safe 的最终 Top-1 排序尚未完全解决”；历史 affine 的 `51/64` 只能作为
非部署诊断上界，不能替代正式模型结果。

下一步不再继续修补“当前 static ranker”，但仍要继续解决 `seq03_small` 所代表的通用小目标
排序问题。新实验必须结构上不同于 lane-wide boost、gain replay 和无保护的 score residual，
并继续禁止使用 target 序列名、帧范围、GT 或 target-derived 阈值。完成这一次有边界的小目标
实验后，再统一训练/推理入口、导出最终 checkpoint、进行轻量化和真正未知序列验证。
自适应协方差 EKF/卡尔曼观测器继续保留给大论文后半部分。

### 10.17 2026-08-06 全实验整理、未闭环问题与修订后的阶段顺序

#### 10.17.1 全部已完成实验的证据矩阵

以下矩阵只保留对当前问题定义有因果价值的证据；同类失败实现、逐 epoch 数字和历史命令保留在对应小节，不再作为主结论。

| 证据组 | 归纳结论 | 对当前问题的作用 |
|---|---|---|
| 采样与候选审计 | S14 的小目标候选覆盖不足，但 anchor 不缺；高分辨率 S7 可将 fixed small R@100 补到 `64/64` | 排除“继续加 anchor/盲目扩大 RPN”作为根本解 |
| fixed target-dev 归因 | correct candidate 已在 14 个失败帧的 top-100 内，但未升至 Top-1 | 将问题定位到最终质量排序 |
| exact-retention 实验族 | affine、lane、static/unified ranker 的 source 原始收益均伴随 old-correct loss；safe 模型则无 small Top-1 增益 | 安全约束不是可放宽的超参数，而是部署前提 |
| 时序与 student | 可维持 source safety、候选覆盖或 far 增益，但 small Top-1 仍为 `50/64` | 不将其他模型的 far gain 误记为 small 解决 |
| support audits（risk、paired-view、cross-fit） | 既有 source-train 没有分散的 real/sim gain 支持，也没有 viable real hard holdout | 当前根因是监督/验证支持不足，不是 head 复杂度不足 |
| 系统审计 | DINO teacher 延迟 `1.74--1.98 s/frame`，最终泛化与统一部署均未完成 | 结构有效后才进入统一、蒸馏和真实未知序列验证 |

#### 10.17.2 当前四个系统级未闭环问题

1. **训练/推理流程未统一。** 当前阶段依赖多个训练、审计入口和结果 JSON 传递 checkpoint
   provenance；尚无一个入口完成原始图像到最终 OBB，也没有一份自包含部署 checkpoint。
2. **DINO 与最终检测模型未形成训练闭环。** 冻结 DINOv2 是当前实验的合法方法边界，
   但目前只有大 teacher + 分阶段轻量 head，没有统一可训练 student、特征蒸馏或最终导出流程。
3. **实时性不达标。** 三张 GTX 1080 上仍需 `1.74--1.98 s/frame`；DINO 前向是主瓶颈，
   仅优化 NMS、S7 ranker 或后处理不能达到实时要求。
4. **最终泛化证据不足。** 固定三切片只用于统一评估，当前部署配置不做 slice routing；
   但这些帧已多次参与诊断，不能继续充当未知域 final test。历史结果 JSON 和 manifest 也必须
   从最终推理依赖中移除。

除上述四项外，当前仍有一个必须先解决的核心检测问题：`seq03_small` 所代表的未知小目标
正确候选虽已进入 top-100，却尚不能稳定排到第一；而 protocol-30--32 已进一步说明，现有
source split 又不足以训练并留出验证一个跨 real/sim 的安全接管器。因此下一阶段的前提是补齐
真实 source hard-sequence 支持，而不是继续修改当前 selector。

#### 10.17.3 修订后的优先级：先改进通用小目标，再做系统统一

在统一训练入口和轻量化之前，原计划允许做一次**有边界、结构不同**的小目标实验；该计划
已经以 native-protected selective promotion V1 实际执行，结果和失败裁决见 10.17.4--10.17.5。
下列内容保留为实验设计与停止门记录，不表示 V1 仍待运行：

```text
固定 native S14 + S7 top-100 候选池
  -> source-only candidate-level quality mean + uncertainty/risk
  -> 与 native top-1 做成对的 conservative advantage 判断
  -> 只有 S7 lower-confidence-bound 明确超过 native upper-confidence-bound 才允许接管
  -> 其余帧精确回退 native
```

这一路线定位为 `native-protected selective promotion`，与已经失败的 lane-wide boost、统一
`+2` replay、non-positive lane suppression 和无保护 static residual 不同。训练只能读取
source GT，使用通用尺度/模糊/亮度扰动和按 source sequence 分组的验证；推理不能把
sequence identity 当作模型特征或特殊路由，也不能读取 `seq03_small` 范围、target GT 或
target-derived threshold。若后续复用因果状态，只允许使用序列边界做 reset。

预注册顺序和停止门为：

```text
第一门：source-only
  old-correct lost = 0
  full >= 688/738
  small >= 311/350
  full/small MCML <= 3
  DFR/ACI 不退化
  按 source sequence 分组验证不能只靠单一序列获益

第二门：一次固定 target-dev（仅第一门通过后）
  seq02_far >= 39/40，MCML <= 1
  seq02_dark >= 29/33，MCML <= 1
  seq03_small >= 51/64，MCML <= 6，R@100 = 64/64
```

这里的 `seq03_small >=51/64` 只作为一次冻结诊断门，不参与训练或阈值选择，也不能替代未来
未知序列的泛化验证。如果 source 门失败，或 target-dev 仍为 `50/64`，立即停止该方向，
不得继续根据这 64 帧修改特征、margin 或 risk threshold。

完成上述一次小目标实验后，后续顺序固定为：

```text
A. 若仍有必要，仅允许结构不同的通用小目标 native-protected V2；不得重复 V1
B. 统一训练/推理入口 + 自包含最终 checkpoint
C. DINOv2-L/14 teacher -> 轻量 student 蒸馏与速度优化
D. 真正未知序列泛化验证
E. 冻结模型的一次性完整 final test，统一导出常规指标及
   R_center / DFR / TDR_w10 / ACI / MCML
F. 自适应协方差 EKF/卡尔曼观测器（大论文后半部分）
```

因此，现在可以整理全部**已有实验**，但不能把论文实验章节写成最终完成状态。当前准确定位是：
候选生成、合并和排序诊断已完成；下一步先解决通用小目标安全接管，然后才进入工程闭环、
轻量化、未知域泛化和最终测试。

#### 10.17.4 native-protected selective promotion V1 已实现并完成 source-only 训练

2026-08-06 已按上述预注册方案完成代码实现，随后在服务器完成了锁定的 source-only 训练。
权威结果由服务器导出并迁移为 `/Users/mac/Downloads/Copy of Copy of train_result.json`；
本轮没有读取 target，也没有进行 target-dev 或完整 test。新增内容为：

- `crane_project/utils/s7_temporal_association.py`：增加 native-vs-S7 成对 advantage mean / uncertainty
  head；用 `LCB = advantage - lambda * uncertainty` 做保守接管，未超过固定 margin 时精确回退
  native top-1。
- `crane_project/tools/dino_teacher_rotated_labeller.py`：增加独立
  `s7_selective_promotion` 训练模式；只训练新 pair head，冻结 DINO、native/S7 proposal、ROI、
  affine 和 phase-2 candidate quality teacher。
- `crane_project/tools/dino_teacher_s7_selective_promotion_train.py`：增加锁定的 source-only
  入口，只接受 phase-2 source-gated epoch-4 checkpoint/result，不读取 target。
- `crane_project/configs/crane_symeood_scoped_dino_lowlight_s7_selective_promotion_v1.py`：记录
  固定结构、loss、inference 与 gate。

选模继续要求 `lost=0`、full/small 绝对门、MCML 门、DFR/ACI 不退化；另新增按
`split|sequence` 统计的多序列净增益门，至少两个 source validation sequence 必须各自净提升，
避免收益只来自单一视频。模型输入不包含 sequence identity、target 切片名或手工问题类型。

实现验证已通过相关单元/回归测试。训练结果表明四个 epoch 都安全回退到 native，正式状态为：

```text
implementation = complete
server_source_training = complete
source_gate = failed
source_best_epoch = 0
selective_s7_promotion = 0 for epochs 1..4
target_dev = not_authorized
formal_deployable_model = native S14 alpha=0.5, S7 disabled
```

#### 10.17.5 2026-08-07 selective promotion V1 代码审计与交接状态

本轮先完成代码收尾和记忆交接，随后补充完成服务器 source-only 实验。代码审计结果如下：

```text
implementation = complete
native_missing branch = patched and unit-tested
py_compile = passed
pytest = 137 passed in 1.69s
git diff --check = passed
server_source_training = complete
source_gate = failed
target_dev = not_authorized
full_test = not_run
formal_deployable_model = native S14 alpha=0.5, S7 disabled
```

代码审计确认：

- `native_protected_selective_promotion()` 在 native lane 缺失时返回
  `selected_index=None`、保持原始候选顺序、`promoted=False` 和
  `reason='native_missing'`，不会把第一个 S7 候选误报为合法输出；
- `LCB == promotion_margin` 使用 `>=`，边界值的行为已经有测试覆盖；
- 训练只更新 `s7_selective_promotion_head.*`，DINO、native/S7 proposal、ROI
  classifier/regressor、global affine 和阶段二 candidate-quality teacher 均冻结；
- 推理只允许 S7 top-100 候选参与成对比较，无法通过 LCB 门时精确回退 native top-1；
- 训练入口锁定 `seed=0`、阶段二 epoch-4 source-gated checkpoint、source-only 和
  `--skip-target-eval`，不接受 target-dev、sequence identity、手工切片路由或 gain replay。

服务器实际运行使用的阶段二输入为：

```text
/home/omnisky/workspace/symEOOD/work_dirs/dino_teacher_s7_temporal_relative_quality_v1/train_result.json
/home/omnisky/workspace/symEOOD/work_dirs/dino_teacher_s7_temporal_relative_quality_v1/labeller_epoch_04_source_only.pth
```

本轮训练使用了新的服务器目录
`/home/omnisky/workspace/symEOOD/work_dirs/dino_teacher_s7_selective_promotion_v2/`，
结果中的 `target_dev=null`、`target_read=false`、`target_used_for_training=false` 和
`target_used_for_checkpoint_selection=false` 均确认没有把三段 target-dev 切片用于训练或选模。

四个 epoch 的 source-only 结果完全相同：

| epoch | full Top-1 | small Top-1 | full/small R@100 | lost | S7 promotion | source gate |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |
| 2 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |
| 3 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |
| 4 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |

失败项是 `full_top1_absolute`、`small_top1_absolute` 和 `multi_sequence_net_gain`；
`exact_old_correct_retention`、MCML、DFR 和 ACI 安全条件没有失败。该结果不是训练崩溃，
而是 LCB 接管门在当前 source gain 正样本极少的监督下没有授权任何 S7 接管，模型因此等价于
“永远保留 native top-1”。这证明了 fallback 的安全性，但没有证明 selective promotion 能
改善 `seq03_small`；因此不能把本轮结果误写成 target 泛化结果。

当前 V1 已完成裁决，后续顺序不再重复 V1：

```text
关闭 selective promotion V1，保留 native S14 alpha=0.5 正式基线
  -> 若继续解决 seq03_small，只允许结构不同的 source-only V2
  -> V2 source gate 通过后一次冻结 target-dev
  -> 统一训练/推理入口与自包含 checkpoint
  -> 轻量 student / 实时性优化
  -> 真正未知序列泛化
  -> 完整 final test 与 R_center/DFR/TDR_w10/ACI/MCML 汇总
  -> 大论文后半部分 EKF/卡尔曼观测器
```

本轮没有改变正式部署模型，没有读取 target，也没有运行完整 test。`docs/*.md` 和
`.workbuddy/memory/*.md` 被 `.gitignore` 忽略；迁移到新账号时仍需单独复制，或显式 force-add
所需 Markdown 文件。

#### 10.17.6 三段 fixed target-dev 的简化边界

这三段只是固定诊断切片，不能拼成一个模型的总成绩，也不是未知序列或部署测试。以下只保留
能界定当前问题的结果；不同方法的历史数字明确隔离。

| 切片 | 当前 highres ROI（protocol-25） | 不可合并的历史参照 | 保留结论 |
|---|---|---|---|
| `seq02_far`，40 帧 | `38/40 -> 38/40`，R@100=`40/40` | temporal student 曾为 `38/40 -> 39/40`，是另一 source-safe 模型 | 当前 highres 未提高 Top-1；far 不再是小目标主线的候选生成瓶颈 |
| `seq02_dark`，33 帧 | `29/33 -> 29/33`，R@100=`33/33` | 无可替代的 Top-1 正增益 | 暗光片段作为保护性诊断；不能称为已完全解决 |
| `seq03_small`，64 帧 | `50/64 -> 50/64`，R@100=`55/64 -> 64/64` | global affine 曾为 `51/64`，但 source `lost=1`，不可部署 | 候选覆盖已补足，正确候选仍未被安全排到 Top-1，是当前唯一明确的研究瓶颈 |

因此，当前唯一可写的跨域诊断结论是：**S7 已解决 `seq03_small` 的候选覆盖，但没有解决最终
质量排序。** 三段均不能替代未知序列泛化、完整系统指标或部署证据。

#### 10.17.7 服务器 `work_dirs` 清理边界

本轮 source-only 结果和历史 target-dev 诊断已经写入本总账；服务器上的大体积
`feature_cache/` 不属于论文证据本身。清理时应保留正式 baseline、阶段二 source-gated
checkpoint、阶段三 target-dev 结果、static ranker 的 source-only 结果和本次 selective
promotion V1 的 `train_result.json`，其余失败路线只在已保留 JSON/日志后删除。不得删除
`dino_teacher_scoped_lowlight_v1_formal8/`、`dino_teacher_fc_cls_interpolation_v1/`、
`dino_teacher_s7_temporal_relative_quality_v1/`、`dino_teacher_s7_temporal_student_v1/`、
`dino_teacher_s7_temporal_student_target_v1/`、`dino_teacher_s7_static_domain_ranker_v1/`
或 `dino_teacher_s7_selective_promotion_v2/`，除非已经另行迁移其 checkpoint 和结果 JSON。

服务器上建议先执行预览，不要直接凭截图中的排序删除：

```bash
cd /home/omnisky/workspace/symEOOD

obsolete_dirs=(
  dino_teacher_s7_quality_suppression_v1
  dino_teacher_s7_lane_arbitration_v1
  dino_teacher_s7_lane_arbitration_v2
  dino_teacher_s7_retention_merge_v1
  dino_teacher_s7_residual_v1
  dino_teacher_roi_cls_pairwise_v2_formal4_v1
)

for name in "${obsolete_dirs[@]}"; do
  path="work_dirs/$name"
  if [ -d "$path" ]; then
    du -sh "$path"
    find "$path" -maxdepth 1 -type f -print
  fi
done
```

确认这些目录中的 `train_result.json` 或 checkpoint 已经迁移、且不再需要原始缓存后，再执行：

```bash
cd /home/omnisky/workspace/symEOOD

obsolete_dirs=(
  dino_teacher_s7_quality_suppression_v1
  dino_teacher_s7_lane_arbitration_v1
  dino_teacher_s7_lane_arbitration_v2
  dino_teacher_s7_retention_merge_v1
  dino_teacher_s7_residual_v1
  dino_teacher_roi_cls_pairwise_v2_formal4_v1
)

for name in "${obsolete_dirs[@]}"; do
  path="work_dirs/$name"
  if [ -d "$path" ]; then
    rm -rf -- "$path"
  fi
done
```

这组命令只针对已经关闭且不属于当前正式基线、阶段二、阶段三、static ranker 或 selective
promotion V1 证据链的目录。照片中名称被截断的 attribution、immediate-override、association
和 quality-attribution 目录暂不纳入自动删除，除非先确认其结果 JSON 已单独保存；它们可能仍是
递归时序审计的原始证据。删除前可先只清理已关闭目录中的 `feature_cache/`，保留 JSON、日志和
checkpoint：

```bash
cd /home/omnisky/workspace/symEOOD
for name in dino_teacher_s7_quality_suppression_v1 \
            dino_teacher_s7_lane_arbitration_v1 \
            dino_teacher_s7_lane_arbitration_v2 \
            dino_teacher_s7_retention_merge_v1 \
            dino_teacher_s7_residual_v1 \
            dino_teacher_roi_cls_pairwise_v2_formal4_v1; do
  find "work_dirs/$name" -type d -name feature_cache -prune -print
done
```

### 10.18 2026-08-08 `seq03_small` 轻量排序 V2 文献依据与预注册方案

#### 10.18.1 当前唯一立即目标

> 本节 10.18.1--10.18.6 是当时的预注册方案和文献依据，保留它们用于追溯设计动机；其后的
> protocol-30--32 已证明当前 source split 缺少分散的 real/sim selector 监督。当前行动边界以
> 第 4 节和 10.18.16--10.18.18 为准，不再启动同类 selector 训练。

下一轮应继续处理 `seq03_small` 所代表的**通用极小目标最终排序问题**，不是重新扩大 S7
候选生成，也不是同时新增 dark expert。当前证据已经足够区分三段困难切片：

```text
seq02_far:   highres 为 38/40；temporal student 历史为 39/40（不可合并）
seq02_dark:  highres 为 29/33，R@100=33/33   -> 保护性诊断，无 Top-1 增益
seq03_small: 50/64, MCML=6, R@20=63/64,
             R@100=64/64                     -> 候选存在，但 Top-1 排序未解决
```

`seq03_small` 的短边约为 `1.124` 个 DINO token；14 个终端失败中，历史逐帧归因有 13 个属于
ROI ordering/NMS，只有 1 个属于 regression。因此下一步必须回答的不是“能否继续增加候选”，
而是：在不破坏 native 旧正确帧的前提下，能否根据**候选相对质量和轻量时序运动残差**，只让
极少数真正更好的 S7 候选接管 native top-1。

`10.18.1--10.18.6` 是在实现前冻结的论文依据、结构、gate 和停止条件。当前本地实现状态见
`10.18.7`；截至该节记录时，**尚未运行服务器训练、尚未读取 target**。

#### 10.18.2 2025--2026 年论文证据与适用边界

下表采用 evidence--claim map：每篇论文只支持其实际覆盖的设计选择，不把点跟踪、世界模型、
光流或卫星视频检测直接等同于本项目的 OBB 候选排序。

| 论文 | 可借鉴证据 | 对本项目的直接用途 | 不能直接推出的结论 |
|---|---|---|---|
| [Chrono: Exploring Temporally-Aware Features for Point Tracking，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Exploring_Temporally-Aware_Features_for_Point_Tracking_CVPR_2025_paper.html) | 在冻结 DINOv2 表征上增加轻量 temporal adapter，使静态基础特征获得视频时序辨识能力 | 支持保留现有 DINOv2，不增加第二个 backbone；只在候选级加入轻量时序预测/适配 | Chrono 是点跟踪方法，不能直接证明其 adapter 会提升旋转目标检测 Top-1 |
| [Back to the Features: DINO as a Foundation for Video World Models，arXiv 2025](https://arxiv.org/abs/2507.19468) | 在冻结 DINO 特征空间中学习未来特征/动态，而不是重新训练大型视觉编码器 | 支持在 frozen DINO ROI embedding 上学习短时运动残差，避免重复 DINO 前向 | 该工作是视频世界模型，不是 native--S7 候选接管器 |
| [Temporally Consistent Object-Centric Learning by Contrasting Slots，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.html) | 使用冻结的 DINOv2 逐帧特征，通过对象级对比目标学习时序一致性 | 支持在 source 视频上增加对象级一致性/对比监督，而不是使用序列名称或固定帧段规则 | 无监督对象中心 slot 与有监督 OBB 排序任务不同，不能原样移植网络和损失 |
| [Optical Flow Estimation for Tiny Objects，IJCAI 2025](https://www.ijcai.org/proceedings/2025/136) | 专门讨论 tiny object 的运动估计，并报告小于 `1%` 的额外参数开销 | 支持“极小目标需要专门的运动残差”，并为轻量化设计提供参照 | 当前不据此引入完整光流网络；先用 box/embedding 的常速度残差验证最低成本版本 |
| [Learn Temporal Consistency for Robust Satellite Video Detector，arXiv 2026](https://arxiv.org/abs/2606.15112) | 将时序特征聚合、结构编码和一致性约束用于遥感视频中的细粒度旋转目标 | 与本项目旋转、小尺寸、跨帧尺度/角度/结构变化最接近；借鉴中心、尺度和周期角度 residual | 完整特征聚合模块更重，且数据域不同；不能把其结果当成本项目的迁移证据 |
| [LSTT: Long-Short Frame Temporal Transformer for Video Object Detection，Expert Systems with Applications 302 (2026)](https://doi.org/10.1016/j.eswa.2025.129921) | 通过长短时特征和自适应 query 利用视频上下文 | 作为后续长短时信息消融的上界参考 | 完整 Transformer 会增加显存、延迟和实现复杂度，不作为当前第一版实现 |

这些论文共同支持的是一个收缩后的结论：**继续复用冻结 DINOv2，以极小的候选级时序模块
补足静态排序，而不是增加条件化双专家、第二套 backbone、光流主干或跨帧 Transformer。**
其中最直接的设计组合是 Chrono/DINO-world 的 frozen-feature 思路，加上 tiny-object motion
和旋转视频检测工作中的结构化运动残差；论文并未现成给出 native--S7 安全接管器，因此接管
门、source exact-retention 和 abstention 仍是本项目自己的核心实验问题。

#### 10.18.3 预注册的轻量 small-object selective ranker V2

V1 失败不是候选消失，而是 source gain 正样本极少，LCB 门最终选择了零接管。V2 必须改变
信息结构，不能只增加 epoch、放宽同一个阈值或重复 V1。推荐结构为：

```text
冻结 DINOv2 + native S14/S7 + ROI heads
                  |
       native top-1 + S7 top-20
                  |
     existing candidate-quality 初筛
                  |
       最可能的一个 S7 候选
                  |
  tiny pairwise advantage/uncertainty head
                  |
  高置信且低风险：S7 接管
  其他情况：严格回退 native
```

第一版只使用约 `16--24` 个标量输入、一个 `hidden_dim=16` 的 MLP，并输出
`advantage` 与 `uncertainty`。不得重复执行 DINO 前向，也不得保存 feature-map history。
候选特征限定为：

- 目标短边的 DINO token 尺度、ROI 面积和长宽比；
- native/S7 分类分数、score margin、entropy 和 candidate-quality 差；
- native/S7 的中心、尺度、角度和长宽比差；
- 基于前两帧状态得到的常速度中心预测残差；
- `log(width)`、`log(height)` 预测残差和周期角度预测残差；
- 当前候选与历史对象的 DINO ROI embedding cosine similarity；
- previous-frame rotated IoU 和候选来源标记。

状态只保存前两帧的 box、ROI embedding 和简单速度量，不新增 dense optical flow、第二个
DINO 模型、跨帧 Transformer 或 dark expert。非小目标、历史不足、uncertainty 高或接管证据
不足时，输出必须与 native 完全相同。

训练仍只在 source 视频上构造 native--S7 pair：

```text
S7 的 source RIoU 明显高于 native：       gain pair
native 已正确且 S7 更差：                 retention pair
两者接近、标注/排序证据不稳定：           abstain pair
```

可使用 balanced pair sampler 或 focal/class-balanced weighting 缓解 gain pair 极少的问题，
但不得无限复制少数正样本，也不得让同一视频序列同时进入训练和验证。source validation 应按
sequence holdout 汇报；最终 gate 使用原始、未重加权的逐帧指标，不能用 sampler 后的训练
分布代替真实分布。

#### 10.18.4 泛化边界和禁止项

该模块是 source-only 离线训练模块，部署时不在线更新。对于相同抓斗、相机、焦距和安装位置
下的新未知视频，不应逐序列重新训练；只有摄像头、镜头、安装位姿、抓斗类型或成像链发生
显著变化时，才考虑离线重校准或域适配。

训练、选模和推理中禁止输入：

- `seq03_small`、`seq02_dark` 等切片名称或 sequence ID；
- 固定帧范围、预定义 target manifest 路由；
- target GT、target-derived threshold 或根据三切片结果选择 epoch；
- 针对 dark/small 的人工逐帧 override；
- 阶段二/三历史 JSON 作为部署时必需输入。

阶段二 `train_result.json` 只可作为 checkpoint provenance 和一次性 source 训练配置的审计
输入；最终统一 checkpoint 必须把所需权重与配置固化，未知序列推理不能依赖该 JSON。

#### 10.18.5 预注册 gate

第一阶段只运行 source training/source validation：

```text
硬安全门：
  lost = 0                         # 相对 native alpha=0.5 的 677 个旧正确帧
  full >= 688/738
  small >= 311/350
  full/small MCML <= 3
  DFR/ACI 不退化

机制有效性门：
  S7 promotion > 0
  至少两个 held-out source sequences 出现净收益
  source-small 上存在可重复净收益
  收益不能只来自单一序列或少数重复 gain pairs
```

只有上述 source gate 全部通过，才授权**一次**冻结的 target-dev 诊断：

```text
seq02_far   >= 39/40, MCML <= 1
seq02_dark  >= 29/33, MCML <= 1
seq03_small >= 51/64, MCML <= 6
seq03_small R@100 = 64/64
```

`51/64` 只是证明相对 `50/64` 有开发进展，不代表未知序列泛化已经解决；真正完成仍要求统一
推理入口后，在从未参与设计和阈值选择的新小目标连续视频上保持 source-safe 并取得提升。

#### 10.18.6 停止条件与后续顺序

- 如果 source `promotion` 仍为 `0`，说明 V2 仍无法从 source 监督中识别安全接管，不再通过
  增加 epoch 重复同一实验。
- 如果 `lost > 0` 或其他 source 硬安全门失败，不读取 target。
- 如果 source gate 通过但一次 target-dev 仍为 `seq03_small=50/64`，关闭当前候选接管路线，
  不围绕这 64 帧继续调阈值；论文中如实报告候选覆盖与排序之间的上限差距。
- 如果达到 `>=51/64` 且 far/dark 不退化，则冻结 V2，进入统一训练/推理入口和自包含
  checkpoint；随后再做轻量 student、速度优化、真正未知序列验证和完整 final test。
- 自适应协方差 EKF/卡尔曼观测器仍保留给大论文后半部分，不提前用于掩盖当前 detector/ranker
  的静态与短时排序问题。

因此，下一次代码修改应只实现“两帧常速度 residual + tiny pairwise
advantage/uncertainty + native abstention”这一版，不再同时优化 `seq02_dark`，也不引入双专家。

#### 10.18.7 已实现、待服务器 source-only 验证

2026-08-08 已完成上述 V2 的本地代码实现，但尚不能宣称该方法有效。新增锁定入口：

```text
crane_project/tools/dino_teacher_s7_small_temporal_ranker_train.py
```

主要实现位于：

```text
crane_project/tools/dino_teacher_rotated_labeller.py
crane_project/utils/s7_temporal_association.py
```

对应测试位于：

```text
tests/test_dino_teacher_rotated_labeller.py
tests/test_s7_temporal_association.py
tests/test_dino_teacher_s7_small_temporal_ranker_train.py
```

新入口启用 `--s7-selective-two-frame`；未启用该 flag 时，历史 selective promotion V1 的结构、
checkpoint 和 gate 行为保持不变。V2 冻结 phase-2 candidate-quality head，并从 S7 lane 的
top-20 候选中只选出一个 quality 最优候选，与 native top-1 构成 pair。tiny head 的输入为 24
个标量，隐藏层维数固定为 16，输出 pairwise advantage 和 uncertainty：

```text
静态相对质量：
  native/S7 score logit 与 margin                  3
  native/S7 frozen quality 与 quality margin       3
  pair center/scale/periodic-angle 差异             3
  native/S7 当前 ROI appearance cosine             1

两帧常速度残差：
  native center/scale/periodic-angle/appearance     7
  S7 center/scale/periodic-angle/appearance         7
                                                   --
总计                                               24
```

接管使用保守 lower-confidence-bound：只有 advantage 扣除 uncertainty 后仍越过 promotion
margin，且 source 监督表明 S7 明显优于 native 时才允许接管；历史不足、帧不连续、候选缺失、
不确定性过高或证据不足时均明确 abstain，保持 native。状态只保留最近两帧**实际输出的 selected
candidate** 的 box 和 ROI embedding；训练和推理使用相同的状态更新语义。周期角度按 OBB 的
`pi` 周期计算最短残差。序列名只用于检测边界并 reset 状态，不进入学习特征。

该实现不增加 DINO、RPN 或 ROI 前向，不保存 dense feature-map history，不引入 optical-flow
网络、跨帧 Transformer、dark expert、EKF 或卡尔曼滤波。所有学习监督仍只来自 source GT；
target、target GT、target-derived threshold、固定帧范围和困难切片名称均不参与训练、选模或
推理。

为避免视频监督泄漏和因抽取 small 帧而破坏因果上下文，本轮同时锁定以下数据逻辑：

- temporal V2 的 source 训练记录固定按 `split, sequence, frame` 排序，不再随机打散相邻帧；
- train/validation holdout 按稳定的 sequence 名称检查，同名序列不能跨 split 同时出现；
- source-small 指标从完整、连续 source validation 的逐帧结果中切片统计，不单独抽帧重跑；
- phase-2 `train_result.json` 仅用于验证 source gate、checkpoint provenance 和固定配置，不提供
  target 样本，也不是未来部署推理的输入；
- phase-2 checkpoint 缺少 V2 tiny head 时只允许初始化该新 head；V2 resume 会严格核对
  `selective_two_frame`、24 个 scalar channels、hidden=16 和 max-candidates=20。

除了原有 source exact-retention gate，V2 还记录并强制检查 selector 本身的实际贡献：

```text
same_frame_set
selector_top1_loss_zero
selector_top1_gain_nonzero
selector_small_top1_gain_nonzero
selector_gain_multi_sequence
```

这些指标只统计 `s7_selective_promotion.promoted=true` 的帧，避免把 phase-2 checkpoint 自身的
收益误记为 V2 selector 的收益。该 selector-specific 强 gate 只在 V2 flag 开启时生效，不会
回写或改变历史 V1 的正式裁决。

本地检查结果：

```text
py_compile:      passed
pytest:          151 passed in 1.02s
git diff --check passed
```

当前状态仍是“代码完成、等待服务器 source-only 裁决”，不是“实验成功”。正式部署 checkpoint
仍为 native S14、ROI classifier `alpha=0.5`、S7 disabled。服务器第一轮只运行 source
training/source validation：若 promotion=0、selector 在 source-small 无实际增益、任何旧正确帧
丢失或其他预注册 gate 失败，均不读取 target；只有全部 source gate 通过后，才授权一次冻结的
三切片 target-dev。若届时 `seq03_small` 仍为 `50/64`，关闭该路线，不围绕这 64 帧继续调参。

#### 10.18.8 V2 服务器结果与下一阶段：stride-7 ROI 空间质量读出

本轮 V2 结果文件显示，训练和 gate 逻辑本身正常，但方法没有学会安全接管：4 个 epoch
均保持 source full `677/738`、small `303/350`、`lost=0`、full/small R@100=`738/738`、
`350/350`，而 `s7_selective_promotion.promoted=0`，selector 自身的 source-small gain
也为零。因此 `best_epoch=0` 是正式 gate 的正确裁决，不是训练日志或 best-epoch 代码错误。
该结果只证明候选覆盖仍在，不能证明 V2 改善了 `seq03_small` 的最终排序；target-dev 未读取。

新发现的问题是：

```text
S7 候选能够进入候选集合，但 V2 只把 S7 lane top-20 预筛成一个候选，
再用 24 个标量判断是否接管；source 中可学习的 gain pair 极少，
因此模型在不确定时始终 abstain，无法利用候选内部的空间细节。
```

因此关闭“标量两帧 selective promotion V2”，但不关闭 `seq03_small` 问题本身。下一轮只做
结构不同的阶段 A：

- 冻结 DINOv2、native S14 RPN/ROI、S7 readout/RPN 和 affine merge；
- 从已存在的冻结 S7 stride-7 feature map 增加一次轻量 `3x3` output rotated RoIAlign readout，
  仅处理 native top-1 加 S7 lane top-32 候选；
- 用 native semantic ROI embedding、stride-7 spatial ROI embedding、score/geometry/source
  标量训练 candidate max-RIoU 与 same-frame relative ranking；
- 推理时只允许经过显式 quality margin 的 S7 候选接管，其他情况保持 native；
- 不增加 DINO forward，不保存 dense feature history，不引入 RGB/前景分支、时序状态、
  双专家、EKF 或卡尔曼；这些属于后续阶段，只有阶段 A source gate 成立后才考虑。

新增锁定入口：

```text
crane_project/tools/dino_teacher_s7_highres_roi_ranker_train.py
```

该阶段仍要求 formal source train/validation、source exact-retention、full `>=688`、small
`>=311`、MCML `<=3`，并且只允许一次 source-only 训练。source gate 未通过不读取 target；
即使阶段 A 通过，也只授权一次冻结的三切片 target-dev。若 source gate 通过但 target-dev
仍为 `seq03_small=50/64`，则关闭当前排序路线，转入统一训练入口和论文中的“候选覆盖已解决、
最终排序仍为开放问题”证据整理，不再围绕这 64 帧调阈值。

本轮实际本地检查为：highres ranker 新增单元测试、历史 V1/V2 回归测试共 `153 passed`，
相关 `py_compile` 通过，`git diff --check` 通过。在服务器 source-only 结果产生前，不把该阶段
称为有效或可部署。

#### 10.18.9 stride-7 ROI ranker 服务器结果：当前最佳 source-safe 实验候选

阶段 A 已完成一次服务器 source-only 训练。训练流程、冻结范围和 target 隔离均符合预注册
设计：`protocol_version=23`，DINOv2 与原检测头参数保持不变，只训练
`s7_highres_spatial_projection` 和 `s7_highres_candidate_quality_head`，可训练参数共
`38,465`；`target_dev=null`，target 未用于训练、checkpoint 选择或评估。

最佳历史输出出现在 epoch 3，epoch 4 与其完全相同：

```text
formal native baseline: full 677/738，small 303/350，MCML 3/3
epoch 1:               full 685/738，small 309/350，lost 0
epoch 2:               full 686/738，small 309/350，lost 0
epoch 3:               full 687/738，small 310/350，lost 0，gained 10
epoch 4:               full 687/738，small 310/350，lost 0，gained 10
```

epoch 3/4 的 full/small R@100 分别达到 `738/738`、`350/350`，MCML 保持 `3/3`；
full DFR 从 `8.4636%` 改善到 `8.3282%`，ACI 从 `0.82687` 改善到 `0.82837`。
S7 成为输出 top-1 的帧数达到 `32`，说明新的 stride-7 ROI 空间质量读出已经实际利用
S7 候选，不再是上一轮 selective promotion V2 的 `promoted=0`。

该结果覆盖此前“最佳 source-safe S7 实验候选”的记录：它相对正式 native 基线新增
`10` 个正确帧和 `7` 个 source-small 正确帧，同时保持 `677/677` exact retention；也比
历史 affine `688/311、lost=1` 更安全。但它不覆盖正式部署模型：正式模型仍为 native S14
`alpha=0.5`、S7 disabled，因为预注册 absolute gate 仍有两项各差 1：

```text
full_top1_absolute:  687 < 688
small_top1_absolute: 310 < 311
```

因此 `best_epoch=0` 和
`SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ` 是现有正式 gate 的正确裁决，
不是 best-epoch 错误。epoch 3/4 checkpoint 只能作为冻结诊断候选，不能使用
`labeller_best_source_only.pth` 代替；后者仍是 epoch-0 fallback。

下一步不重复训练，只对固定 epoch 3 checkpoint 做一次 source-only、共享模型前向的
promotion-margin 审计。预注册 margin 为 `0.20、0.225、0.25`；每帧只计算一次
DINO/S7/ROI/quality logits，然后在同一候选输出上离线应用三个 margin。选择规则不变：
`lost=0`、full `>=688`、small `>=311`、full/small MCML `<=3`、DFR/ACI 不退化；若多个
margin 通过，固定数值最大的 margin。该审计仍不读取 target。只有审计通过，才授权一次
固定三切片 target-dev；否则不继续围绕 source 的 1 帧反复训练或调参。

该审计现已实现，锁定入口为：

```text
crane_project/tools/dino_teacher_s7_highres_margin_source_audit.py
```

实现额外强制：输入结果必须精确匹配 `protocol_version=23` 的 epoch 3
`687/310、lost=0、gained=10` near-pass；checkpoint 必须是同一 work-dir 下的
`labeller_epoch_03_source_only.pth`；`0.25` margin 必须在本次只读审计中复现 epoch 3，
否则立即停止。三个 margin 复用同一帧的 frozen DINO、S7、ROI 和 high-resolution quality
logits，只重算最终接管判断。审计输出使用 `protocol_version=24`，不更新参数、不复制
checkpoint、不读取 target。另已修复 fallback 结果中
`current_inference_small_validation_summary=null` 的纯报告遗漏；该遗漏不影响既有训练或 gate。

服务器 margin audit 已完成，结果满足预注册 source gate：

```text
margin 0.20:  full 688/738，small 311/350，lost 0，gained 11，promotion 35
margin 0.225: full 688/738，small 311/350，lost 0，gained 11，promotion 34
margin 0.25:  full 687/738，small 310/350，lost 0，gained 10，复现 epoch 3
```

三组结果来自同一批 `738` 次 frozen model forward 和 `2214` 次 margin decision；没有重新
训练或更新参数。按“通过正式 source gate 的最大 margin”规则，固定选择 `0.225`。该结果的
full/small MCML 均为 `3`，full DFR 为 `8.3185%`，ACI 为 `0.82870`；相对 native 的
`677/303` 保持 `677/677` exact retention，并新增 `11` 个 full、`8` 个 small 正确帧。
因此该 epoch 3 + runtime margin `0.225` 是当前第一个通过正式 source gate 的 S7 实验候选，
但 target 尚未读取，正式部署模型仍不改变。

下一步仅授权一次固定三片段 target-dev。已新增独立只读入口：

```text
crane_project/tools/dino_teacher_s7_highres_fixed_target_audit.py
```

入口要求输入 `protocol_version=24` 的 source margin 结果，并同时锁定：checkpoint epoch 3、
checkpoint 架构 margin `0.25`、source 选出的运行时 margin `0.225`。这样既不伪造 checkpoint
结构，也不允许在 target 上再次搜索 margin。三片段仍固定为 `seq02_far[2..41]`、
`seq02_dark[137..169]` 和 `seq03_small[129..192]`，仅作为统一策略的评估集合；片段名称、
sequence ID 和帧范围不进入模型特征或推理路由。正式 target-dev gate 为：

```text
seq02_far   >= 39/40，MCML <= 1
seq02_dark  >= 29/33，MCML <= 1
seq03_small >= 51/64，MCML <= 6，R@100 = 64/64
```

脚本还逐帧核对候选 checkpoint 中的 native top-1 是否与正式 native baseline 完全复现，
报告 Top-1、MCML、R@20、R@100、mean RIoU、DFR、ACI、gained/lost frames，并再次检查 DINO、
baseline head 和 candidate head 参数未改变。它不执行 optimizer step，不增加 DINO forward，
不做 target-derived threshold 或片段特化。只有三片段 gate 全部通过，才授权完整 test；即使
通过，也不能直接宣称未知序列泛化或部署完成。

第一次启动服务器 fixed target audit 时，在读取 target 前被 checkpoint provenance gate 拦截：
历史 epoch 3 checkpoint 的嵌套 `s7_highres_roi_ranker` 元数据没有保存冗余的
`frozen_detector=True` 标记，而初版审计脚本误把该标记当作必需字段。该错误只发生在元数据
校验阶段，三片段尚未开始推理，不产生 target 结果，也不影响 checkpoint 权重。

现已修复为历史 schema 兼容规则：嵌套审计标记缺失时，由 protocol-24 source 审计中的
`dino_parameters_unchanged`、`detector_parameters_unchanged`、只读零参数更新，以及 checkpoint
的 source-only/frozen-DINO、epoch、training mode 和严格 architecture validation 共同证明；
如果历史 checkpoint 中存在这些嵌套字段，则任何与固定设计冲突的值仍会被拒绝。复合错误也已
拆分为 source isolation、special routing、extra compute 和 shape 四组，后续报错可直接定位。

本地检查：相关 `py_compile` 通过，highres 训练、source margin audit、fixed target audit、
历史 V1/V2 与主 labeller 回归测试共 `165 passed`；`git diff --check` 通过。服务器三片段
target-dev 尚未完成，当前不能提前报告其结果。

#### 10.18.10 2026-08-10 fixed target-dev 新问题与统一排序阶段

本节覆盖 10.18.9 之后已经完成的固定 target-dev 诊断。结果来源为：

```text
/Users/mac/Downloads/Copy of target_dev_result.json
```

该文件的 `protocol_version=25`，裁决为
`S7_HIGHRES_FIXED_TARGET_DEV_FAIL_KEEP_NATIVE_BASELINE`。source margin gate、checkpoint
provenance gate、baseline/native 逐帧复现、参数零更新、DINO 与检测 head 未改变，以及
`candidate_forward=137=40+33+64` 均通过；因此这次失败不是评估索引错位或 target 泄漏。

固定三片段结果如下：

| 片段 | baseline -> candidate Top-1 | MCML | R@100 | 结论 |
|---|---:|---:|---:|---|
| `seq02_far` | `38/40 -> 38/40` | `1` | `40/40` | 未达到预设 `39/40` |
| `seq02_dark` | `29/33 -> 29/33` | `1` | `33/33` | 通过该片段诊断门 |
| `seq03_small` | `50/64 -> 50/64` | `6` | `55/64 -> 64/64` | 候选覆盖通过，Top-1 未改善 |

`seq03_small` 的实际 high-resolution promotion 只有 frame `133、182、186` 三帧，且三帧
原本已经 Top-1 正确，因此没有产生新增 Top-1 gain。14 个失败帧中，正确候选已经在池内，
其候选排名为：rank 2 有 9 帧，rank 3 有 2 帧，rank 8、rank 12、rank 39 各 1 帧。由此
得到的新研究问题不是“stride-7 是否还能产生候选”，而是：

```text
在候选已经覆盖的跨域小目标帧上，如何用 source-only 监督学习一个可靠的 native/S7
统一质量排序，使正确候选从 rank 2/3 等位置安全提升到 Top-1，同时保持 native
exact-retention，而不是让 high-resolution 分支只在已经正确的帧上 promotion？
```

本结果支持论文中的限定性表述：stride-7 高分辨率 ROI 读出在保持 source exact-retention
的同时，将 `seq03_small` 的候选覆盖从 `55/64` 提升到 `64/64`，但最终 Top-1 仍为
`50/64`；跨域瓶颈已经从候选生成转移到候选质量排序。不能据此声称 `seq03_small` 已解决、
highres ranker 已泛化、checkpoint 可部署或三个 target 片段已全部通过。

必须继续区分以下历史结果：global affine 的 `51/64` 带有 source lost=1，不能部署；temporal
student 的 `seq02_far=39/40` 与当前 highres 不是同一模型；这些数字不能合并成一个已完成的
结果。当前正式部署模型仍是
`work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth`，即
native S14、ROI classifier `alpha=0.5`、S7 disabled。

#### 10.18.11 下一阶段：unified native/S7 hard-pair ranker

这次不再重复三片段 target-dev、不在 target 上继续扫描 margin，也不扩大 S7 RPN 或再叠加同构
post-hoc ranker。代码阶段改为在现有 highres ROI 入口中增加显式的
`--s7-highres-unified-ranking` 路径：

- DINOv2、native S14 RPN/ROI、S7 readout/RPN、affine merge 和 OBB 几何保持冻结；
- 对同一个 `native top-1 + S7 top-32` active pool 建立统一 fused score，而不是只监督某个
  lane winner；
- 用 source GT 的连续 max-RIoU、同帧相对排序和 current-fused-score hard pairs 训练，显式
  保留 native retention pair 与 S7 gain pair；
- 推理时所有 active candidates 共用一个 rank score，但 S7 仍必须超过 native 的固定
  promotion margin 才能接管；零初始化时严格回退 native；
- source-only 训练期间增加确定性的 feature-domain brightness/blur/scale view，作为跨视图
  排序稳定性压力，不读取 target、不使用 sequence ID、不引入时序状态、额外 DINO forward、
  RGB/前景分支或 EKF/Kalman；
- 训练、推理和审计复用同一组 quality logits，避免再出现“训练目标与最终接管规则不一致”。

该设计对应 proposal ranking、IoU-guided hard-pair ranking 与 rank/sort 质量排序的文献方向：
[Learning to Rank Proposals for Object Detection (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.html)、
[RankDetNet (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_RankDetNet_Delving_Into_Ranking_Constraints_for_Object_Detection_CVPR_2021_paper.pdf)、
[Rank & Sort Loss (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html)
以及单源 domain-generalized detection 的 view diversification/alignment 方向
[Danish et al. (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Danish_Improving_Single_Domain-Generalized_Object_Detection_A_Focus_on_Diversification_and_Alignment_CVPR_2024_paper.html)。这些论文是设计依据，不是本项目已经取得的实验结果。

新增/修改的实现入口为：

```text
crane_project/utils/s7_temporal_association.py
crane_project/tools/dino_teacher_rotated_labeller.py
crane_project/tools/dino_teacher_s7_highres_roi_ranker_train.py
```

本阶段服务器只允许先运行一次 source-only 训练和 source gate。若 source exact-retention、
full/small absolute gate、MCML 或 frozen-component isolation 任一失败，立即保留失败证据，
不读 target；只有新的 source gate 通过后，才设计与其 architecture/protocol 对应的一次新
固定 target-dev 审计。该新 checkpoint 在此之前不是正式部署模型，也不能用于未知序列泛化
claim 或完整 final test。

#### 10.18.12 unified 训练结果与两级 source 裁决

unified source-only 训练已经完成，`target_dev=null`。四个 epoch 都只训练 high-resolution
projection/quality head，DINO 与原检测 head 参数保持不变。正式 exact-retention gate 的
`best_epoch=0` 不是 checkpoint 保存错误：epoch 1--4 均损失同一帧
`val|real_seq07|215`，因此 `exact_old_correct_retention=false`。原始 source 指标为：

| epoch | full Top-1 | small Top-1 | lost | gained | full/small MCML |
|---:|---:|---:|---:|---:|---:|
| 1 | 688/738 | 312/350 | 1 | 12 | 3/3 |
| 2 | 695/738 | 319/350 | 1 | 19 | 3/3 |
| 3 | 696/738 | 320/350 | 1 | 20 | 3/3 |
| 4 | 695/738 | 319/350 | 1 | 19 | 3/3 |

其中 epoch 3 是 raw source 最优点，但不能回写成原 exact-retention 协议通过。exact retention
不是所有目标检测研究的通用指标；它是本项目为“替换当前正式模型”预先设定的风险约束。
因此后续采用两级裁决：

1. `formal exact gate` 保持不变，继续决定 `source_safe` 和正式部署候选资格；
2. 新增 `bounded-risk research gate`，只允许在 source full/small、MCML、DFR、ACI 均满足原
   gate 的其余要求时，容许 `lost<=1`、lost fraction `<=0.002`、gain/loss ratio `>=10`，
   且净收益至少跨两个 source sequence。它最多授权一次新的固定 target-dev 诊断，不能授权
   部署、完整 final test 或 source-safe claim。

为避免立刻重训或在 target 上调阈值，先锁定 epoch 3 checkpoint，用同一组 frozen source
forward/logits 审计固定 margin `0.25/0.275/0.30`。如果更保守 margin 能恢复 `lost=0` 且
保留正式 full/small gate，则按原 exact gate 选择；否则只按上述 bounded-risk 字段判断是否
值得继续一次固定诊断。该审计仍不读取 target、不更新参数，也不重复已有三片段实验。

实现入口：

```text
crane_project/tools/dino_teacher_s7_unified_highres_margin_source_audit.py
crane_project/tools/dino_teacher_rotated_labeller.py
tests/test_dino_teacher_s7_unified_highres_margin_source_audit.py
```

#### 10.18.13 protocol-26 source margin 结果与 Pairwise Takeover V2 依据

结果文件：

```text
/Users/mac/Downloads/margin_audit_result.json
```

本次 `protocol_version=26` 审计是**正向但受限的 source-only 结果**。它完整复现 epoch 3，
`target_dev=null`；DINO 与检测器参数保持不变，参数更新为 0，738 个 source 帧共享一次模型
forward，并产生 `738*3=2214` 个固定 margin decision。三个运行时 margin 的结果完全相同：

| runtime margin | full Top-1 | small Top-1 | lost/gained | full/small MCML | exact gate | bounded-risk gate |
|---:|---:|---:|---:|---:|---|---|
| 0.25 | 696/738 | 320/350 | 1/20 | 3/3 | fail | pass |
| 0.275 | 696/738 | 320/350 | 1/20 | 3/3 | fail | pass |
| 0.30 | 696/738 | 320/350 | 1/20 | 3/3 | fail | pass |

相对 native source baseline，full Top-1 从 `677` 增至 `696`，small Top-1 从 `303` 增至
`320`，full R@100 从 `717` 增至 `738`，small R@100 从 `329` 增至 `350`；DFR 从
`0.08464` 降至 `0.07871`，ACI 从 `0.82687` 增至 `0.83910`。收益跨越 real/sim 两个
source sequence，但净收益明显偏向 sim：`real_seq07 +2`，`sim_seq10 +17`。唯一 source
损失仍为 `val|real_seq07|215`，lost fraction 为 `0.001477`，gain/loss ratio 为 `20`。
因此结果可作为“unified high-resolution 排序在 source 上显著提高候选覆盖与净 Top-1”的
正向消融证据，也可报告 exact-retention 与 aggregate gain 之间的安全权衡；但不得写成
source-safe、target 改善、未知序列泛化或部署结果。当前裁决仍为：

```text
SOURCE_ONLY_UNIFIED_HIGHRES_BOUNDED_RISK_RESEARCH_GATE_PASSED_TARGET_NOT_READ
source_safe=false
eligible_for_fixed_target_dev_diagnostic=true
eligible_for_deployment=false
eligible_for_full_test=false
```

三个 margin 的 promotion 数、Top-1、lost/gained 帧和时序指标均完全相同，说明
`0.25--0.30` 没有跨过新的排序边界；继续扫描全局 margin 不能解决该错误接管。0.30 只是
source 指标打平后的保守 tie-break，不代表它优于 0.25。另需保留 raw Top-1 `696/320`
与 deployment-threshold Top-1 `694/318` 的差异，正式部署前必须增加逐帧 deployment
exact-retention gate。

Pairwise Takeover Ranker V2 后续已经完成 source-only 训练，但没有通过 source gate；其关键
结果与失败原因仅保留在 10.17.1 测试总表中，不再展开不可行路线。target 未读取，正式模型
仍为 native S14、ROI classifier `alpha=0.5`、S7 disabled。

#### 10.18.14 protocol-27 smooth-geometry support audit：信号为正，但 small 覆盖门控过严

结果来源：`/Users/mac/Downloads/result.json`。本次审计没有读取 target，参数更新为 0，
738 个 source 帧共享一次 forward，native baseline 逐帧复现通过。它不是训练结果，也不应
被解释为部署或 target 泛化结果。

审计发现的关键事实是：`full` 上 `sym_kld` 达到 `732/738`、`55` 次 gain、`0` 次 loss；
`source-small` 上达到 `348/350`、`45` 次 gain、`0` 次 loss。`gwd` 与
`normalized_gwd` 也在 full/small 上有正净收益。因此几何排序信号并非为零。

原 protocol-27 之所以给出
`SOURCE_ONLY_SMOOTH_GEOMETRY_RANK_SUPPORT_INSUFFICIENT_TARGET_NOT_READ`，是因为它对
`full` 和 `small` 使用同一个 `min_gain_domains=2`、`min_gain_sequences=2`。当前正式
source-small 划分只有 `sim` 域和 `sim_seq10` 一个序列，这两个条件在该子集上结构性不可
满足，属于 coverage gate 设计不匹配，不是模型 forward 或评估索引错误。

本轮已将审计协议升级为 protocol-28：full 仍保持 `2/2` 覆盖要求；small 使用独立的
`small-min-gain-domains=1`、`small-min-gain-sequences=1`，同时 JSON 显式记录
`coverage_limited=true`。这只允许进入一次 source-only 研究训练，不允许宣称 small
多域/多序列泛化、source-safe、部署或完整 final test。

同时新增几何引导统一排序训练选项：以 source GT 计算训练期 smooth OBB geometry
（默认 `sym_kld`）并形成辅助 pairwise ranking loss；推理时仍只使用 native/S7 候选特征
生成的 quality logits，不读取 GT/target，不改变 DINO、native detector、S7 candidate
pool 或 native-protected promotion contract。入口为：

```text
crane_project/tools/dino_teacher_s7_highres_roi_ranker_train.py
--smooth-geometry-ranking
```

训练前必须先获得 protocol-28 support JSON；support gate 未通过时训练入口会拒绝启动。
本次代码修改尚未运行服务器训练，当前正式部署模型不变。

#### 10.18.15 protocol-29 native-relative risk residual：有效 penalty 被 gate 锁死，target 未读取

本轮 `dino_teacher_s7_native_relative_risk_v2` 的结果**有记录价值，但仅作为实现/监督失效机制证据，不是性能正结果**。它在统一 high-resolution quality head 上加入“仅压低错误 S7、保护可用 S7”的 native-relative risk residual；训练和选模均为 source-only，三个固定 target-dev 片段均未读取。

| epoch | source full / 738 | source small / 350 | lost / gained | source gate |
|---|---:|---:|---:|---|
| 1 | 696 | 320 | 1 / 20 | fail |
| 2 | 696 | 320 | 4 / 23 | fail |
| 3 | 697 | 322 | 4 / 24 | fail |
| 4 | 699 | 323 | 4 / 26 | fail |

唯一失败项始终是 `exact_old_correct_retention`；`best_epoch=0`，`source_selected_checkpoint` 指向 epoch-0 native fallback（不是 S7 候选），结果裁决为 `SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ`，因此 `target_dev=null`，不具备 target-dev、未知序列泛化、部署或完整 final test 的资格。

这次定位到的机制比“训练损失没有下降”更具体：风险 head 的 raw `risk_scale` 在 epoch 1 已变为 `-1.381873e-05`，epoch 2--4 分别约为 `-1.378079e-05`、`-1.377574e-05`、`-1.377574e-05`。现有 one-sided 参数化计算 `max(raw_risk_scale, 0)`，负区间导数为零，因而 effective `risk_scale`、risk penalty mean/max/nonzero 全程为零；训练实际等价于未启用 risk residual 的统一 ranker。日志中的 risk-retention active count 并非不存在，但仅约 10、12、11、16 个/epoch，而每 epoch 约有 2,781 个 source train frame、约 81k 个错误 S7 候选，监督极度稀疏。

结论：

- 不可把 699/323 当作可选 checkpoint；它以 4 个旧正确帧损失换得 gains，违反 exact retention。
- 不可据此宣称 native-relative risk 思路无效；本实现的 gate 已死锁，不能产生任何风险 penalty。
- 也不应仅把 `max(raw,0)` 换成可导函数后直接复训同一方案：正向 active 风险对极稀疏，先前 source gate 的失败机制仍未被结构性解决。
- 下一步限定为**只读 source-only native-relative risk / score-gap support audit**：在冻结 epoch-0 native fallback 上临时恢复已加载的 S7 candidate readout，仅量化 native-correct/wrong-S7 对、required penalty 与 score-gap 的分布、跨 domain/sequence 覆盖和 oracle S7 的排名。该 audit 不训练、不选 checkpoint、不读取 target、不产生 deployment 阈值，也不改变 checkpoint 或正式 S7-disabled 部署；只有其证明存在足够且分散的 source 支持，才讨论结构性 V2（正参数化、gain-balanced pair sampling 和 native-preserve 约束）。

当前正式部署模型仍保持 `work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth`：native S14 + ROI classifier alpha=0.5，S7 disabled。

#### 10.18.16 protocol-30 native-relative risk support audit：有效 retention 正监督为零，关闭该路线

只读 source-only 审计 `Source-only Native-relative Risk Score-gap Support Audit` 已完成。审计使用 protocol-29 的 epoch-0 native fallback，在进程内临时启用已冻结的 S7 candidate readout；共执行 `2,781` 次 source train/train_sim candidate forward，候选数固定为 `33/frame`。`parameter_update_count=0`，DINO 和检测 head 参数均未改变，`target_dev=null`，没有 target 训练、选模或阈值调节。

主要统计：

| 指标 | 结果 |
|---|---:|
| source train frames | 2,781 |
| native-correct frames | 2,774 |
| wrong S7 candidates | 82,291 |
| usable S7 candidates | 6,701 |
| native-wrong / usable-S7 frames | 7 |
| 以上 gain frames 的覆盖 | 仅 sim / `sim_seq08` |
| oracle 为 S7 的 frames | 652 |
| oracle S7 未被 base fused 排为 Top-1 | 601 |
| effective risk penalty nonzero | 0 |

JSON 表面记录 `active_required_penalty_candidate_count=1`、`required_penalty=0.138764`，但该唯一候选来自 `train_sim|sim_seq08|74`：native 已错误（RIoU=`0.370908`），同帧有 2 个 usable S7。protocol-29 的 retention objective 只在 `native_correct and wrong_s7` 时启用，因此该候选不属于 V1 可训练的 retention 正监督。`active_required_penalty_frame_count=0` 正是这一事实的反映。当前汇总字段把 all-frame required penalty 与 native-correct eligible retention support 混在同一层，容易误读；后续应在 schema 中显式分开，但现有 `frame_rows` 已足以重算，不需要再次执行 2,781 帧 forward。

正式裁决：

- epoch-0 上 native-relative risk V1 的 eligible positive retention support 为 `0`；protocol-29 后期每 epoch 约 10--16 个动态 active pair 是 quality head 漂移后才出现的极稀疏信号，不能证明该 residual 有稳定的初始可学习支持。
- 7 个真实 takeover gain frame 全部来自一个 simulation sequence，缺少 real 域与多序列覆盖；不能据此训练或声称未知序列泛化。
- risk residual 只能压低 S7，不能解决这 7 个“native 错、正确 S7 需要上升”的 gain case；监督方向与剩余问题不一致。
- protocol-29 native-relative risk residual V1 正式关闭；不进行简单 gate 改写后复训，也不进入同目标 V2，不读取固定 target-dev，不运行完整 final test。
- 该结果是有用的负证据：候选覆盖仍然存在，但当前 source 分布没有为“仅抑制错误 S7”提供足够且跨域的正监督。下一阶段必须转向 source-only 的跨域 candidate-quality/order 学习或 source 数据支持扩展，并排除已失败的 PQA/QFL、lane suppression、static/temporal quality、Pairwise Takeover、smooth-geometry 与 margin scanning 路线。

当前正式部署模型继续保持 `work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth`：native S14 + ROI classifier alpha=0.5，S7 disabled。

#### 10.18.17 protocol-31 paired-view candidate role-switch support audit：跨域支持不足，target 未读取

protocol-30 说明原始 source 中 native-wrong/S7-correct 的 gain frame 只有 7 个，且全部位于
`sim_seq08`；因此不能直接继续训练同类 S7 接管 head。protocol-31 已按如下入口完成：

```text
crane_project/tools/dino_teacher_s7_paired_view_role_switch_support_audit.py
```

它锁定 unified high-resolution epoch-3 candidate checkpoint。对每个 source-train/train-sim 原始帧，
分别运行 `clean`、`photometric`（gamma/exposure/contrast）和 `degradation`
（blur/noise/downsample-upsample）三个**图像级**视图；每个视图都重新经过冻结 DINO、native/S7
RPN 和 ROI，不能复用或扰动 cached feature tensor。变换不含 crop、flip、rotation 或其他空间 warp，
故原始 OBB 标注保持有效；feature cache key 同时固化 view 名称与版本。source validation 只运行
clean view，并必须复现锁定 native baseline full `677`、small `303` 后，才允许解释 train 的支持统计。

协议不训练、不选 checkpoint、不读取 target、不产生部署阈值，且将 DINO/检测 head 参数版本在前后比较。
同一原始帧在多个增强视图中产生的信号只计为一个 unique support frame。预注册训练授权门为：

```text
paired clean-native -> augmented native-wrong/S7-correct role-switch frames >= 32
gain role-switch 覆盖 real 和 sim
gain role-switch 覆盖 >= 3 个 source sequences
任一 sequence 占 role-switch frames <= 50%
native-retention support 同时覆盖 real 和 sim
```

实际结果为 `SOURCE_ONLY_PAIRED_VIEW_ROLE_SWITCH_SUPPORT_INSUFFICIENT_TARGET_NOT_READ`：

| 指标 | 结果 |
|---|---:|
| clean source-val reproduction | full `677/738`，small `303/350`，通过 |
| candidate forwards | `9,081 = 2,781 × 3 + 738` |
| parameter updates | `0` |
| paired gain role-switch 原始帧 | `10 / 32` |
| role-switch 覆盖 | 仅 sim / `sim_seq08` |
| 单一 sequence 占比 | `1.0`，要求 `<=0.5` |
| retention support | `0` |
| target-dev | `null` |

两种 augmented view 并非未生效：photometric/degradation feature cache 均为首次计算，且
degradation 将 train native hit 从 `2,774` 降至 `2,768`、native-wrong/S7-correct 从 `7` 增至
`13`。10 个 role-switch 都是 `sim_seq08` 的临界 RIoU 帧：clean native 刚好正确，而增强后 native
跌破 `0.5`、S7 仍正确。因此它们是单一模拟序列中的人工退化支持，不能作为跨域排序训练集。

该结果同时发现 split 层面的监督错配：protocol-31 train 包含 `real_seq01/04/05/06` 与
`sim_seq08`，其中 real train `2,033` 帧全部 native 正确；clean validation 却有 `57` 个
native-wrong/S7-correct frame，其中 `real_seq07=9`、`sim_seq10=48`。不能把这些 validation frame
直接并入旧训练集，否则会破坏现有 source gate；也不能放宽 protocol-31 coverage gate 或继续扩大
degradation。

#### 10.18.18 protocol-32 source sequence cross-fit support feasibility audit：当前 split 无 real held-out 支持

下一步新增 JSON-only 入口：

```text
crane_project/tools/dino_teacher_s7_sequence_crossfit_support_audit.py
```

它仅读取 protocol-31 `train_frame_rows` 与 `validation_clean_frame_rows`，不读取图像、DINO、
checkpoint、target 或标注，也不进行参数更新。它按完整 sequence 枚举 leave-one-sequence-out
fold，明确禁止随机 frame split；每一 fold 的训练侧必须同时有 real/sim natural gain 支持，held-out
侧必须保留足够的困难 gain frame。正式总 gate 还要求至少一个 viable real hard held-out sequence
和一个 viable sim hard held-out sequence，防止只在 simulation sequence 上训练/验证。

protocol-31 的结果 JSON 已经由该 JSON-only 审计复核，结果为
`SOURCE_ONLY_SEQUENCE_CROSSFIT_SUPPORT_INSUFFICIENT_TARGET_NOT_READ`，无 DINO forward、无参数更新、
`target_dev=null`。7 个 sequence 的 leave-one-sequence-out 中，有 `2` 个 viable fold：held-out
`sim_seq08` 与 held-out `sim_seq10`；但 `real_seq07` held-out 时，训练侧只剩 simulation natural
gain，故 `viable_heldout_real_sequences=[]`。总 gate 的 `minimum_valid_folds=true`、sim held-out
coverage=true，但 real held-out hard-sequence coverage=false。

这说明目前 source 数据无法构成无泄漏的跨域 selector 训练/验证协议；关闭 learned S7 selector，
后续应补充至少两个独立标注的真实 source 困难连续序列（其 frozen native 错误、S7 有正确候选），
并按完整 sequence 留出验证，而不是使用 target、把 `real_seq07` 直接并回旧训练集，或调低门槛。
当前正式部署模型不变。

#### 10.18.19 protocol-33 dense object-centric temporal separability：source-only 支持门通过

protocol-32 只否定了依赖稀疏 `native-wrong/S7-correct` 接管标签的现有 selector 训练协议；它
没有否定利用所有连续 source 标注帧学习对象级时序表示。`seq02_dark` 的有效改进也说明，方法
可以通过改变监督任务而不依赖分支接管标签：其冻结 DINO 检测头使用的是密集 RPN/ROI source
监督，而不是 native/S7 takeover supervision。

因此新增 protocol-33 入口：

```text
crane_project/tools/dino_teacher_s7_dense_temporal_separability_audit.py
```

它锁定 unified high-resolution epoch-3 candidate checkpoint，在 source `train + train_sim` 的
完整连续序列上，对 `clean/photometric/degradation` 三个图像级视图分别重新运行冻结 DINO、
native/S7 RPN 和 ROI。每帧使用 source GT 将 active pool 候选标为：`max-RIoU>=0.5` 的对象正
候选，以及 `max-RIoU<=0.30` 的高分 hard negative。随后只读统计：

- 相邻帧正确候选相对于高分错误候选的 ROI embedding / highres embedding cosine margin；
- clean 与两个 label-preserving view 之间的对象级 cosine margin；
- 使用前两帧正确候选构造的中心、对数尺度和周期角度常速度残差可分性；
- 按完整 source sequence 分组的 pair 数和成功比例。

该设计与已失败的阶段三 temporal student 不同：student 训练同帧连续 RIoU、relative quality
和 teacher distillation，并在固定七标量递归选择器上推理；protocol-33 不训练 selector，而是先
判断 frozen ROI/highres 表示是否在多个真实 source 连续序列中具备对象级跨帧、跨视图可分性。
它不会把 sequence identity 输入模型，也不读取 target、更新参数、选择 checkpoint 或生成部署
阈值。

预注册默认支持门为：每个 sequence 同时具有至少 `32` 个 temporal pair 和 `32` 个 cross-view
pair；二者在固定 cosine margin `0.02` 下的成功比例都不低于 `0.60`；满足条件的 source sequence
至少覆盖 `2` 个 real 和 `1` 个 sim。运动残差完整报告但暂不作为硬门，避免在未验证 camera/crane
运动尺度前用手工权重误杀有效外观信号。只有 protocol-33 通过，才允许实现 source-only 的轻量
temporal contrastive candidate adapter；否则关闭现有数据上的时序表示学习并转向补充真实 source
hard sequence。无论结果如何，都不授权 target-dev、部署或完整 final test。

实际结果为 `SOURCE_ONLY_DENSE_TEMPORAL_SEPARABILITY_PASS_TARGET_NOT_READ`：冻结候选前向
`9,081` 次、参数更新 `0`、target 为 `null`，并复现 clean source-val native baseline full
`677/738`、small `303/350`。source-train 得到 temporal pair `2,773`、cross-view pair
`5,562`；组合外观 margin 成功率为 `0.8990/0.9642`，中位数为 `0.09196/0.11664`，合格
完整序列覆盖 real `seq01/04/05/06` 与 sim `seq08`。

分解结果表明，主要可学信号来自 frozen ROI 1024-D embedding：temporal/cross-view margin
中位数约 `0.1845/0.2322`；原 highres 32-D embedding 的 cosine margin 近零（约
`0.000049/0.00069`）。`real_seq05` 是 real source 中最困难的完整连续序列，因此后续仅作为
内部 sequence holdout，禁止随机拆帧。该结果只授权 source-only adapter 训练，仍未授权读取
固定 target-dev。

#### 10.18.20 protocol-34 ROI temporal contrastive candidate adapter V1：表征门通过，但 source gate 失败

新增入口 `crane_project/tools/dino_teacher_s7_roi_temporal_contrastive_train.py`。它锁定
protocol-33 的 unified high-resolution epoch-3 候选基座，冻结 DINO、native/S7 RPN、ROI 与
原 highres quality head；只训练 `1024 -> 128 -> 64` 的 L2-normalized ROI projector。监督来自
完整 source 连续序列的相邻帧正候选、三个固定视角的跨视角正候选和同帧 top-8 hard negatives，
不再依赖稀疏 takeover 标签。训练使用 real `seq01/04/06` 与 sim `seq08`，完整
`real_seq05` 仅用于内部检索指标选 epoch；官方 source-val 只在 epoch 固定后执行一次最终 gate。
训练后 holdout 的 margin 成功率和中位数还必须同时不低于 epoch-0 原始 ROI cosine；否则即使
后续检测数字偶然通过，也不能把随机投影或退化表示解释为有效学习。
实现进一步锁定 temporal/cross-view 两个分支分别非退化，epoch selection 先最大化二者较弱项，
再依次比较 temporal、cross-view 成功率及两类中位 margin；不能用数量更多的 cross-view pair
掩盖 temporal 退化。

推理先确定 native fallback，仅当最佳 S7 候选相对 native 的 learned ROI cosine 与常速度 OBB
残差融合优势达到固定 `0.05` 时才接管；断帧或 sequence 边界立即复位。正式 gate 保持旧正确帧
`lost=0`、full `>=688`、small `>=311`、full/small MCML `<=3`，并要求 DFR/ACI 不退化；失败时
保留 native baseline，不授权 target-dev、部署或完整 final test。
active pool 的上限语义固定为“额外 `1` 个 native + 最多 `32` 个 S7”，native 不占 S7 名额。
source 结果还必须分别报告 margin promotion 与所有 S7 state update（包括 native 无有效候选时的
S7 fallback）的连续长度、错误状态连续长度、S7 状态后旧正确帧丢失、恢复间隔和未恢复事件；其中
“S7 状态后旧正确帧丢失为 `0`”显式并入 source gate，不增加 target-derived 阈值。

8G 显存约束已写入入口：三个视角逐个前向；仅将代表正候选和 top-8 hard negatives 以 CPU
FP16 保存到 `roi_temporal_source_train_cache_fp16.pt`；正式 adapter batch 固定为 `128`；
legacy SDPA query chunk 最大 `256`；默认每帧调用一次 CUDA cache 释放。该机制限制活跃张量和
缓存上界，但 `nvidia-smi` 的 reserved memory 仍可能随算子 workspace 在安全范围内波动。
正式 protocol-34 进一步锁定最多三张 8G GPU：两张只承载 frozen DINO shard，一张承载冻结
检测 head 与 adapter。该拓扑不是数据并行，adapter 的 data-parallel world size 始终为 `1`，
effective batch 固定 `128`，故学习率固定 `3e-4`，不能按可见 GPU 数量进行线性缩放。正式入口
同时锁定 epochs `4`、temperature `0.07`、promotion margin `0.05`、motion weight `0.25`，避免
资源调整被误写成新的超参数扫描。

实际 source-only 运行选中 epoch `4`。`real_seq05` 内部 holdout 上，adapter 的 temporal/cross-view
检索成功率达到 `0.989247/0.997321`，中位 margin 达到 `0.709181/0.699611`，明显高于原始 ROI
cosine 的 `0.655914/0.839286` 与 `0.07304/0.13698`，故 representation gate 通过。这证明
protocol-33 所发现的对象级时序信号确实可以被轻量 projector 学习，而不是随机投影造成的假象。

但正式 source-val 闭环选择严重失败：native baseline full/small=`677/738`、`303/350`；候选结果
仅为 `425/738`、`135/350`，full/small MCML=`203/125`，old-correct lost=`263`、gained=`11`。
共发生 `381` 次 S7 promotion，其中 `302` 次错误；最长 S7 状态连续段为 `286` 帧，最长连续错误
promotion 为 `203` 帧，S7 状态之后发生 `261` 个旧正确帧丢失。失败集中于 `sim_seq10`：错误 S7
被写入下一帧 reference 后，错误轨迹在外观相似度与运动一致性上自强化。裁决为
`SOURCE_ONLY_ROI_TEMPORAL_ADAPTER_FAIL_KEEP_NATIVE_BASELINE`；target 为 `null`，不授权固定
target-dev、部署或完整 final test。

#### 10.18.21 protocol-35 ROI temporal counterfactual attribution：确认状态污染，但否定直接接管准则

为区分实现闭环错误与表征假设本身，新增只读 source-only 反事实审计：固定 protocol-34 epoch-4
adapter、promotion margin `0.05`、motion weight `0.25`，在同一次冻结候选前向中比较
`closed-loop/native-anchor` 与 `top-32/top-8` 四种策略；参数更新为 `0`，target 为 `null`，不进行
margin scan。`native-anchor` 允许 S7 改变当前输出，但下一帧状态始终由 native fallback 更新，因而
不会让错误 S7 递归污染长期 reference。

结果如下：

| 策略 | source full | source small | lost/gained | full/small MCML |
| --- | ---: | ---: | ---: | ---: |
| native baseline | `677/738` | `303/350` | `0/0` | `3/3` |
| closed-loop top-32 | `425/738` | `135/350` | `263/11` | `203/125` |
| native-anchor top-32 | `658/738` | `284/350` | `35/16` | `8/8` |
| closed-loop top-8 | `415/738` | `144/350` | `274/12` | `130/130` |
| native-anchor top-8 | `663/738` | `290/350` | `31/17` | `5/5` |

native-anchor 将 S7 state update 与 post-S7-state old-correct loss 都降为 `0`，同时把 full 从
`425` 恢复到 `658/663`，因此确认 protocol-34 的主要灾难性退化来自递归状态污染。然而最佳
native-anchor top-8 仍比 native baseline 少 `14` 个 full Top-1、少 `13` 个 small Top-1，并丢失
`31` 个旧正确帧；top-8 也没有稳定优于 top-32 的闭环证据，故候选池范围失配不是根本原因。

当前最准确的机制裁决为：

> Protocol-33 正确证明了 source ROI 表征具有对象级时序可分性；Protocol-35 进一步证明，这种
> 可分性不足以直接构成 native--S7 takeover quality criterion。其主要原因是对象身份/轨迹一致性
> 与候选相对几何质量并不等价：连续错误背景同样可以形成稳定时序表征。递归状态隔离能够消除长链
> 漂移，却不能解决单帧错误接管。

因此关闭基于 ROI temporal cosine 的同构 takeover selector，不继续重训、扫描 margin 或读取
target。该结论不否定 protocol-33 的表征证据，而是限定其适用范围：它可支持对象级关联表示，
不能在缺少可验证 real source takeover supervision 时被直接解释为安全的 native--S7 质量排序器。
正式部署模型仍为 native S14、ROI classifier `alpha=0.5`、S7 disabled。

#### 10.18.22 SymEOOD--DINO 统一检测 config：实现完成，等待联合 source gate

此前最新 native S14 `alpha=0.5` checkpoint 主要通过 labeller 与独立审计入口运行；单独将其
包装成纯 DINO config 也不能代表论文中的 SymEOOD 主方法。现已新增标准 MMRotate 联合配置：

```text
crane_project/configs/crane_symeood_dino_unified_v1.py
```

该配置保留正式 `crane_symeood_k1.py`：SymNFLLoss、SymKLDLoss、SymPOLA 与 K=1 辅助监督
训练出的主头先产生一个 SymEOOD top-1 OBB。该 OBB 被转换到 DINO 输入尺度，作为额外 proposal
与 frozen DINO native-S14 RPN proposals 合并；所有候选再统一经过同一个 source-safe DINO ROI
分类/回归头，最终只输出一个 top-1 OBB。SymEOOD 原始 score 不参与融合，从而避免直接比较两套
未校准分数。这既不是纯 DINO，也不是两个检测结果的启发式二选一。

DINO 分支固定加载 `work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth`；
MMRotate 位置 checkpoint 则加载
`work_dirs/crane_symeood_k1/best_Weighted_R_center_epoch_12.pth`。原始 SymEOOD 日志按 source-val
`Weighted_R_center` 选择 epoch12=`0.9945`，高于 epoch24=`0.9932`；后续分支从 epoch24 初始化的
约定不等于论文主模型的 checkpoint 选择。BrightAug、target scope、
sequence-ID routing、S7、temporal takeover 和 box stabilizer 均关闭。DINO checkpoint provenance
contract 仍要求 selector/protocol/alpha、target-unread、source gate、source full/small 摘要和
S7-disabled 状态一致。

该工作形成了真正的联合推理结构，但仍是一个**新的、尚未验证的组合模型**。此前 SymEOOD 与
DINO 的各自数字只能作为组件证据，不能相加或冒充联合模型结果。当前没有重新读取固定三片段
target-dev；必须先运行联合 source gate，确认 old-correct retention、full/small Top-1、MCML、
DFR/ACI 与仿真角度指标不退化，之后才决定是否授权完整 test。后续对象级深度 head 应只接收
该统一前端冻结后的 Top-1 OBB、score 及可选 ROI feature，不得反向改变检测排序。

## 11. 新账号记忆读取顺序与迁移

新账号不要只依赖聊天历史。按以下顺序读取，即可恢复当前项目状态：

1. `docs/innovation4_dino_small_target_progress.md`：本阶段权威实验总账，包含所有已完成
   测试、正式 gate 裁决和唯一下一步。
2. `docs/innovation4_frozen_dinov2_lowlight_large_target.md`：冻结 DINO 暗光大目标分支。
3. `.workbuddy/memory/MEMORY.md`：项目级历史索引；再按需读取同目录的日期文件。
4. `/Users/mac/.codex/memories/MEMORY.md`：当前机器的 Codex 全局记忆索引。
5. `/Users/mac/.codex/memories/extensions/ad_hoc/notes/20260801-symeood-s7-current-handoff.md`：
   本次最新 S7/merge/lane 裁决和新账号接手摘要。
6. `/Users/mac/.codex/memories/rollout_summaries/2026-07-22T11-40-23-2zLw-symeood_s7_lidere_retention_merge_handoff.md`：
   S7 初始实现、被拒绝 checkpoint 与 retention-aware merge 设计证据。
7. `/Users/mac/.codex/memories/rollout_summaries/2026-07-27T02-59-06-b3wr-symood_dino_unified_small_object_diagnosis_literature_2024_2.md`：
   小目标失败类型与论文路线证据。
8. `/Users/mac/.codex/memories/skills/symeood-mcml-diagnosis/SKILL.md`：项目 MCML 诊断流程。

`.workbuddy/memory` 位于项目目录，可随整个工作区迁移；`/Users/mac/.codex/memories`
属于当前 macOS 账号的用户目录，新账号不会自动拥有，必须显式复制需要的文件。另需
注意本仓库 `.gitignore` 的 `*.md` 会忽略本文及其他 Markdown 文档；如果仅通过 Git
迁移，必须显式 force-add 这些文档，或单独复制 `docs/` 与 `.workbuddy/memory/`。

当前项目内全部 `.workbuddy` 记忆文件为：

```text
.workbuddy/memory/MEMORY.md
.workbuddy/memory/2026-07-07.md
.workbuddy/memory/2026-07-08.md
.workbuddy/memory/2026-07-09.md
.workbuddy/memory/2026-07-10.md
.workbuddy/memory/2026-07-13.md
.workbuddy/memory/2026-07-14.md
.workbuddy/memory/2026-07-16.md
.workbuddy/memory/2026-07-17.md
.workbuddy/memory/2026-07-18.md
.workbuddy/memory/2026-07-20.md
.workbuddy/memory/2026-07-21.md
.workbuddy/memory/2026-07-22.md
.workbuddy/memory/2026-07-24.md
.workbuddy/memory/2026-07-27.md
.workbuddy/memory/2026-07-28.md
```

当前账号中与 SymEOOD 直接相关的 Codex 记忆入口还包括：

```text
/Users/mac/.codex/memories/memory_summary.md
/Users/mac/.codex/memories/MEMORY.md
/Users/mac/.codex/memories/raw_memories.md
/Users/mac/.codex/memories/extensions/ad_hoc/notes/2026-06-20-gate-norm-tta-result.md
/Users/mac/.codex/memories/extensions/ad_hoc/notes/2026-07-26-symeood-dino-innovation.md
/Users/mac/.codex/memories/extensions/ad_hoc/notes/20260722-185612-shared-filter-oracle-next.md
/Users/mac/.codex/memories/extensions/ad_hoc/notes/20260723-145450-p3-feature-shift-supported.md
/Users/mac/.codex/memories/extensions/ad_hoc/notes/20260724-100104-multimodal-fpn-closed-dino-teacher-next.md
/Users/mac/.codex/memories/extensions/ad_hoc/notes/20260801-symeood-s7-current-handoff.md
/Users/mac/.codex/memories/skills/symeood-mcml-diagnosis/SKILL.md
/Users/mac/.codex/memories/rollout_summaries/2026-07-22T11-40-23-2zLw-symeood_s7_lidere_retention_merge_handoff.md
/Users/mac/.codex/memories/rollout_summaries/2026-07-27T02-59-06-b3wr-symood_dino_unified_small_object_diagnosis_literature_2024_2.md
```

其余 rollout summaries 是更早实验的会话级证据，由全局 `MEMORY.md` 按任务索引；迁移
时若需要完整历史，直接复制整个 `/Users/mac/.codex/memories/`（排除其中 `.git/` 即可）。

## 12. 2026-09-05 V4、因果历史与 seq11-v2 完整交接入口

本轮从固定 V4、measurement-validity、几何 refiner、dual tower、因果历史 V2/V3、seq11
初始 59 帧到 seq11-v2 251 帧 replay 的完整实验链，已独立汇总到：

```text
docs/20260905_symeood_dino_fusion_seq11_full_handoff.md
```

其中记录了有效与无效路线、source-val/fixed TEST 结果、关键代码与产物位置，以及最新
V4+seq11-v2 `epoch_10` 的证据边界。当前 `epoch_10` 只是官方 738 帧上的暂定多目标折中候选；
DFR 已被纳入稳定性约束，但并非 DFR 单项最优。由于现有 seq11 aux-val 48 帧不含任何困难
支持样本，正式 promotion 与再次 fixed TEST 之前必须先完成 251 帧三折时序块 OOF/CV。
