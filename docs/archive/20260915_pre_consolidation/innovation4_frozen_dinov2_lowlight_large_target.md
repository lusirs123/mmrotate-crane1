# 创新点 4：冻结 DINOv2 语义分支的暗光大目标排序恢复

更新时间：2026-07-26  
项目：SymEOOD / CraneOBB 破坏性环境下的旋转目标检测与控制评价

## 1. 当前结论

暗光分支之后的远距离/小目标归因、分类头更新、多尺度审计和下一步计划，统一记录在：
[`docs/innovation4_dino_small_target_progress.md`](innovation4_dino_small_target_progress.md)。

本方法针对的是 `real_seq02[137..169]` 暗光大目标困难片段中的**分类排序坍塌**，不是普通的亮度增强，也不是远距离候选生成修复。

BrightAug 在 source 上形成了稳定的 `FPN level1 -> anchor0 -> classifier0` 路由；暗光 target 的正确候选通常位于 `FPN level0 -> anchor2`，几何候选仍然存在，但排序分数极低。冻结 DINOv2 分支提供了另一条语义特征路径，使 source 训练的检测头能够在该困难片段中重新把正确目标排到前面。

当前最可靠的证据是：

```text
real_seq02[137..169]
BrightAug:   top-1 = 0/33,  RIoU-MCML = 33
ScopedDINO:  top-1 = 29/33, RIoU-MCML = 1
稳定化DINO: top-1 = 32/33, RIoU-MCML = 1
```

因此，冻结 DINOv2 对暗光大目标的目标发现和排序问题是有效的。但当前 scope 使用了 target-dev 困难片段，因此这是 target-dev/low-light hard-subset 结果，不应包装成完全独立的 zero-shot final test。

## 2. 原始问题和根因

### 2.1 BrightAug 的路由缺口

对 source 训练、target atlas、FPN pathway 和分类卷积的联合审计得到：

```text
source 稳定路径：FPN level1 + anchor0 + classifier0
target 暗光正确路径：FPN level0 + anchor2 + classifier2
```

训练覆盖审计中，source 的 anchor2 没有有效的正几何和分类监督。target 中大多数困难帧仍存在 `RIoU >= 0.5` 的候选，正确候选只是被分类排序推到很深的位置。

因此问题不能简单描述为“暗光导致所有特征变弱”，更准确的描述是：

```text
geometry remains
classification ordering collapses
```

### 2.2 已排除的简单解释

- padding/valid-content 只改善 usable rank，不能恢复 top-1；
- quality-only teacher-forced 只能达到 oracle 上界，不能转化为可训练部署收益；
- 复制 source anchor0 filter 不能识别 target level0 特征；
- guided background proxy 同时破坏分类和几何，不能授权 adapter；
- AdaBN 的统计偏移没有被亮度条件稳定区分，不能作为已验证根因；
- P3 objectness、P3/P4 neighborhood、普通 FPN kNN 审计没有提供普通特征空间迁移证据；
- `real_seq02[2..41]` 的远距离/候选饥饿问题与 137..169 暗光排序问题分开处理。

## 3. 方法设计

### 3.1 核心结构

方法是一个 scope-gated frozen-DINO semantic rescue branch：

```text
输入图像
  ├─ BrightAug 主检测器：通用路径，保持原有能力
  └─ 冻结 DINOv2 ViT-L/14 patch grid
       -> 单尺度 OrientedRPNHead
       -> RotatedRoIAlign 7x7
       -> RotatedShared2FCBBoxHead
       -> 旋转框候选
       -> source-selected 因果旋转框稳定器
            仅平滑宽、高和周期角度
            保持中心、分数、排序和输出存在性不变
```

DINOv2 主干不更新，只训练 RPN/ROI 检测头。推理时不做 score fusion：scope 内 DINO 是 primary，DINO 无输出时才回退 BrightAug；scope 外完全使用 BrightAug。因果稳定器只处理 scope 内最终选中的 top-1 旋转框，不重新选择候选。

### 3.2 和 DINO Teacher 的关系

设计借鉴 CVPR 2025 DINO Teacher 的核心思想：使用冻结的 DINOv2 语义表征作为教师式视觉基础，再训练轻量检测头适配任务。

本项目没有声称复现原论文完整的 target pseudo-label、教师-学生联合训练或全套域对齐流程。当前实现是一个受控的、source-only 的冻结语义分支：

- DINOv2 权重冻结；
- 检测头只使用 source 标注训练；
- 不使用 target pseudo label；
- 不在 target 上做 optimizer step；
- target 标签只在诊断和离线评测阶段使用。

### 3.3 为什么需要因果稳定器

冻结 DINOv2 分支解决的是“目标能否被找到并排到第一位”，但 DINO 的 RPN/ROI
检测头逐帧独立预测。暗光片段恢复输出后，相邻帧的框宽、高和角度可能发生
跳动，使 DFR 上升、ACI 下降。这个现象并不表示 DINO 找错了目标，而是恢复的
同一个目标在不同帧使用了不稳定的旋转框参数。

因此，最终方法将问题拆成两个互补步骤：

```text
冻结 DINOv2 语义分支：恢复目标发现和分类排序
因果旋转框稳定器：    约束已恢复目标的跨帧尺度与角度变化
```

稳定器不参与候选生成和分类，也不改变 DINO 与 BrightAug 的路由决策。

### 3.4 因果状态和输入输出

设第 `t` 帧最终选中的旋转框为：

```text
b_t = (c_x,t, c_y,t, w_t, h_t, theta_t, score_t)
```

稳定器只保存上一帧的稳定化几何状态：

```text
hat_b_(t-1) = (c_x,t-1, c_y,t-1, hat_w_(t-1), hat_h_(t-1), hat_theta_(t-1))
```

第 `t` 帧只使用当前框 `b_t` 和上一帧状态 `hat_b_(t-1)`，不读取未来帧，
所以它是严格的在线因果处理，可以直接用于顺序推理，而不是双向平滑或
teacher-forced 后处理。

稳定器明确保持以下量不变：

```text
hat_c_x,t = c_x,t
hat_c_y,t = c_y,t
hat_score_t = score_t
候选顺序、检测是否存在、DINO/BrightAug 路由均不改变
```

这使稳定器不会凭空补出检测，也不会把原本错误排序的候选提升为 top-1；
排序恢复仍完全来自冻结 DINOv2 语义分支。

### 3.5 旋转框等价表示对齐

同一个旋转矩形可以写成两种等价参数：

```text
(w, h, theta) == (h, w, theta + pi/2)
```

如果直接对两种表示做平均，可能把一个长方形错误地平滑成接近正方形，并
产生约 `90°` 的虚假角度跳变。因此在平滑之前，先在当前框的两个等价表示中
选择与上一稳定框最接近的一个。

选择代价为：

```text
C = || log([w,h]) - log([hat_w_(t-1),hat_h_(t-1)]) ||_2^2
    + Delta_pi(theta, hat_theta_(t-1))^2
```

其中 `Delta_pi` 是考虑旋转框 `pi` 周期后的最短角度差。分别计算
`(w,h,theta)` 和 `(h,w,theta+pi/2)` 的代价，选择代价较小的表示进入后续
平滑。这一步只消除参数表达歧义，不改变旋转矩形的实际几何区域。

### 3.6 尺度和周期角度平滑

宽、高在对数空间使用因果指数滑动平均：

```text
log(hat_w_t) = (1-alpha) log(hat_w_(t-1)) + alpha log(w_t)
log(hat_h_t) = (1-alpha) log(hat_h_(t-1)) + alpha log(h_t)
```

使用对数空间而不是直接平均，是因为尺度变化更接近比例变化，并且能够保证
平滑后的宽、高始终为正值。

旋转框角度具有 `theta` 与 `theta+pi` 等价的周期性，不能直接对角度数值做
线性平均。首先把角度映射到双角单位圆：

```text
v_t     = [cos(2 theta_t), sin(2 theta_t)]
hat_v_t = (1-alpha) hat_v_(t-1) + alpha v_t
hat_theta_t = 0.5 atan2(hat_v_t[y], hat_v_t[x])
```

双角表示把相差 `pi` 的两个等价方向映射到同一点，因此能够跨越角度边界平滑，
避免 `89°` 与 `-89°` 被错误平均到 `0°`。

### 3.7 系数选择和状态重置

候选系数固定为：

```text
alpha in {0.25, 0.5, 0.75, 1.0}
```

`alpha` 越小，历史状态影响越强；`alpha=1.0` 表示完全不修改当前框，是保底
候选。系数只在 official source validation 上选择，target 标签不参与。候选
必须满足：source-val top-1 不下降、MCML 不增加、mean RIoU 下降不超过
`0.005`、ACI 不下降；在合格候选中选择 DFR 最低者，最终得到 `alpha=0.25`。
这表示新状态保留 `75%` 的上一稳定状态并吸收 `25%` 的当前观测。

为了防止历史状态跨越无关片段传播，出现以下任一条件时立即清空状态：

- 当前帧不在 DINO scope 内；
- 当前帧没有检测输出；
- 序列发生变化；
- 帧号不连续。

因此稳定器只在同一低照度片段的连续有效输出之间工作。它改善的是恢复框的
时序几何一致性，不会影响 scope 外 BrightAug 的检测结果。

## 4. 训练协议

配置文件：

```text
crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py
```

### 4.1 数据协议

```text
source train: train + train_sim，2781 帧
source val:   official val，738 帧
target-dev:   real_seq02[137..169]，仅用于训练完成后的诊断
complete test: 992 帧
```

训练阶段没有读取 target 标签，checkpoint 只依据 source-val 选择。target-dev scope 参与了方法设计和路由定义，因此完整 test 结果标记为 diagnosis-only。

### 4.2 DINOv2 配置

```text
模型：DINOv2 ViT-L/14
patch size：14
输入高度：600
最长边：1333
DINO blocks：GPU 1、GPU 2 分片
RPN/ROI head：GPU 0
PyTorch 兼容：legacy SDPA，query chunk=512
```

### 4.3 可训练检测头

```text
OrientedRPNHead：单尺度，stride=14，feat_channels=256
anchor scales：32、64、128、256、512 像素
anchor ratios：0.5、1.0、2.0
RotatedRoIAlign：7x7
RotatedShared2FCBBoxHead：fc=1024，单类别
ROI samples：256
proposal count：2000
max detections：2000
```

### 4.4 独立训练日程

DINO 小头的 epoch 与 BrightAug 的 epoch 没有对应关系。当前正式协议为：

```text
epochs：8
optimizer：SGD
lr：0.001
momentum：0.9
weight decay：1e-4
gradient clip：10
linear warmup：1000 iterations，ratio=0.001
step decay：epoch 5、7，gamma=0.1
checkpoint：每个 epoch 保存并在 source-val 评估
seed：0
```

source-val 选择结果：

```text
best DINO head epoch：8
source-val top-1：662/738
source-val top-1 MCML：7
source-val R@20：662/738
source-val mean top1 RIoU：0.6592
```

BrightAug 仍使用独立的 source-val `ckpt_sweep.py` 选择流程，候选 epoch 为 16、18、20、22、24；本次完整 test 使用 `epoch_20.pth`。

## 5. 验证过程

### 5.1 结构性诊断阶段

在正式训练 DINO 小头之前进行了以下受控审计：

1. **Anchor/FPN coverage**：确认 source 的稳定监督集中于 level1/anchor0，target 暗光正确候选位于 level0/anchor2。
2. **Shared-filter counterfactual**：复制 anchor0 filter 后 target top-1 仍为 0/33，排除“只需增强 anchor0 filter”。
3. **Frozen P3 objectness probe**：source-val 很高，但 target paired margin `0/31`，不能支持普通 P3 objectness 迁移。
4. **Feature alignment audit**：target usable P3 特征在 source whitening、方向和 cosine 审计中全部呈 source-negative-like，强支持 level/feature domain shift 解释。
5. **P3/P4 neighborhood rescue 与 multimodal kNN**：P3/P4 均未通过 source-control gate，关闭普通 FPN 邻域救援路线。
6. **ROI head probe**：冻结 DINO 特征上的 source ROI head 在 source-val 可以稳定学习，但早期 target 结果仍不足以作为正式部署结论。
7. **Temporal beam / rerank**：只能把少量候选提升到 top-1，不能稳定解决整段失败，关闭为主要方法路线。

### 5.2 历史 legacy 诊断

早期曾使用 `val/real_seq07` 按 `frame_id % 5` 划分：181 帧 source-train、45 帧 source-val，得到过 target-dev `32/33、MCML=1`。该结果用于路线探索，但不是正式协议，因为它没有使用 official train/train_sim 和 official val 的正式划分。

正式论文结果应使用 2781 帧 source train、738 帧 source val、8 epoch 的正式 source-only 训练结果，而不是把早期 181 帧 legacy 结果和正式结果混合。

## 6. 暗光专项结果

### 6.1 target-dev 暗光大目标切片

```text
片段：real_seq02[137..169]

方法                         top-1       旋转 IoU-MCML
BrightAug                    0/33        33
ScopedDINO                   29/33       1
```

该结果说明 DINO 分支确实恢复了暗光大目标的分类排序，而不是简单增加随机背景框。

### 6.2 作用范围

完整 test 中：

```text
总帧数：992
DINO scope：33 帧
BrightAug-only：959 帧
target-dev scope：target_label_derived=True
final-test eligible：False
```

因此 sim_seq09 和 real_seq03 没有被 DINO 改动，普通片段的 BrightAug 能力得到保留。

## 7. 完整 test 结果

下面保留项目一直使用的 `CraneOfflineEvaluator` 输出格式，便于和历史实验直接比较。

### crane_symeood_k1+简单数据增强

```text
┌─ [real 域] ──────────────────────────────────────────────
│  第一层  静态精度（单帧）
│    real/R_center(%)                                     98.91
│    real/mean_RIoU                                       0.8251
│  第二层  时序稳定性（帧间）
│    real/DFR(%/frame)                                    2.6441
│    real/ACI                                             0.9364
│  第三层  控制适用性（系统级）
│    real/TDR_w10(%)                                      80.85
│    real/MCML_max(frames)                                39
│    real/MCML_mean(frames)                               25.5
│    real/MCML_pass(limit=5)                              0
│    real/MRF(frames)                                     1.0

┌─ [sim 域] ──────────────────────────────────────────────
│  第一层  静态精度（单帧）
│    sim/A-RMSE(deg)                                      1.5488
│    sim/R_center(%)                                      100.0
│    sim/mean_RIoU                                        0.9204
│  第二层  时序稳定性（帧间）
│    sim/DFR(%/frame)                                     1.8935
│    sim/ACI                                              0.9621
│  第三层  控制适用性（系统级）
│    sim/TDR_w10(%)                                       100.0
│    sim/MCML_max(frames)                                 0
│    sim/MCML_mean(frames)                                0.0
│    sim/MCML_pass(limit=5)                               1
```

### crane_symeood_k1+简单数据增强+冻结DINOv2

```text
┌─ [real 域] ──────────────────────────────────────────────
│  第一层  静态精度（单帧）
│    real/R_center(%)                                     97.39
│    real/mean_RIoU                                       0.8214
│  第二层  时序稳定性（帧间）
│    real/DFR(%/frame)                                    3.8323
│    real/ACI                                             0.9182
│  第三层  控制适用性（系统级）
│    real/TDR_w10(%)                                      88.31
│    real/MCML_max(frames)                                39
│    real/MCML_mean(frames)                               25.5
│    real/MCML_pass(limit=5)                              0
│    real/MRF(frames)                                     1.0

┌─ [sim 域] ──────────────────────────────────────────────
│  第一层  静态精度（单帧）
│    sim/A-RMSE(deg)                                      1.5488
│    sim/R_center(%)                                      100.0
│    sim/mean_RIoU                                        0.9204
│  第二层  时序稳定性（帧间）
│    sim/DFR(%/frame)                                     1.8935
│    sim/ACI                                              0.9621
│  第三层  控制适用性（系统级）
│    sim/TDR_w10(%)                                       100.0
│    sim/MCML_max(frames)                                 0
│    sim/MCML_mean(frames)                                0.0
│    sim/MCML_pass(limit=5)                               1
```

### crane_symeood_k1+简单数据增强+冻结DINOv2+因果框稳定器

```text
┌─ [real 域] ──────────────────────────────────────────────
│  第一层  静态精度（单帧）
│    real/R_center(%)                                     97.39
│    real/mean_RIoU                                       0.8212
│  第二层  时序稳定性（帧间）
│    real/DFR(%/frame)                                    2.6402
│    real/ACI                                             0.9395
│  第三层  控制适用性（系统级）
│    real/TDR_w10(%)                                      88.31
│    real/MCML_max(frames)                                39
│    real/MCML_mean(frames)                               25.5
│    real/MCML_pass(limit=5)                              0
│    real/MRF(frames)                                     1.0

┌─ [sim 域] ──────────────────────────────────────────────
│  第一层  静态精度（单帧）
│    sim/A-RMSE(deg)                                      1.5488
│    sim/R_center(%)                                      100.0
│    sim/mean_RIoU                                        0.9204
│  第二层  时序稳定性（帧间）
│    sim/DFR(%/frame)                                     1.8935
│    sim/ACI                                              0.9621
│  第三层  控制适用性（系统级）
│    sim/TDR_w10(%)                                       100.0
│    sim/MCML_max(frames)                                 0
│    sim/MCML_mean(frames)                                0.0
│    sim/MCML_pass(limit=5)                               1
```

### 7.1 标准 CraneDataset 补充指标

`CraneDataset` 标准评测也应保留，但必须和上面的 legacy 离线指标分开命名：

```text
指标                              BrightAug       ScopedDINO    稳定化DINO
standard mAP                      0.81683         0.81705       0.81750
real_R_center（全 real 帧）        0.6476          0.7238        0.7238
sim_R_center                      1.0000          1.0000        1.0000
Weighted_R_center                 0.8943          0.9171        0.9171
sim_A_RMSE                        1.5487          1.5487        1.5487
```

两套 `R_center` 的含义不同：

- `CraneOfflineEvaluator` 当前完整 test 调用使用 `center_thresh=15px`，且只对有输出的帧统计，因此 `98.91%` 是条件中心精度；
- `CraneDataset` 使用配置中的 real `25px`、sim `10px`，silence 记为失败，`0.6476 -> 0.7238` 是全 real 帧中心召回率。

论文中不应把这两个数写成同一个无说明的 `R_center`。建议命名为：

```text
R_center^det@15px：有输出帧条件中心精度
R_center^all@25px：包含 silence 的全帧中心召回率
```

### 7.2 完整 test 的正确解释

完整 test 中 ScopedDINO 将精确旋转 IoU top-1 命中数从 `809/992` 提升到 `838/992`，但完整 real 域 `MCML_max` 仍为 `39`。原因是：

- real_seq02 的 `2..40` 是远距离候选不足问题，DINO scope 不覆盖；
- real_seq03 的 `129..192` 是小目标几何/候选问题，DINO 没有启用；
- 暗光排序失败区间 `133..171` 被打散，因此 DINO 的局部收益被完整序列 MCML 掩盖。

## 8. DFR/ACI 退化及其修复

未稳定化的 DINO 版本存在时序几何退化：

```text
real DFR: 2.6441 -> 3.8323
real ACI:  0.9364 -> 0.9182
```

原因是 DINO ROI head 逐帧独立预测，暗光片段中预测框的宽、高和角度出现明显跳动；同时 DINO 填补了 BrightAug 原先的 silence，使更多相邻框变化进入 DFR 统计。

只使用 source validation 选择的因果稳定器已经修复该问题：

```text
ScopedDINO real DFR:       3.8323 -> 2.6402
ScopedDINO real ACI:       0.9182 -> 0.9395
BrightAug real DFR/ACI:    2.6441 / 0.9364
稳定化后 top-1/MCML:       32/33 / 1
scope 外变化帧:            0
```

因此最终创新点应表述为：

```text
冻结 DINOv2 恢复暗光大目标的语义发现与分类排序，
source-selected 因果框稳定器进一步恢复连续输出的几何稳定性。
```

这两个问题由两个互补部件处理：冻结 DINO 分支负责重新找到并排序目标，
因果稳定器负责约束恢复框的跨帧尺度和角度变化。

## 9. 数据泄露和声明边界

### 没有发生的泄露

- target-dev 没有进入 DINO 小头训练；
- target-dev 没有参与 source checkpoint 选择；
- 没有 target optimizer step；
- 没有 checkpoint 写入 target-derived 权重；
- DINOv2、RPN/ROI head 参数在完整 test 前后均保持不变；
- 959 个非 scope 帧完全沿用 BrightAug。

### 仍然存在的声明限制

`real_seq02[137..169]` 参与了困难问题定位、scope 设计和方法验证，因此不能称为完全未见的独立 final test。论文应分开报告：

1. source-val：用于 DINO head checkpoint 选择；
2. target-dev low-light hard subset：证明暗光大目标排序恢复；
3. full test：报告整体组合效果和非回归情况；
4. `real_seq02[2..41]`、`real_seq03[129..192]`：作为独立的远距离/小目标失败分析，不与暗光排序问题合并。

## 10. 可直接放入论文的结果文字

### 方法效果

> 为缓解暗光大目标场景中的分类排序坍塌问题，我们引入冻结 DINOv2 ViT-L/14 语义分支，并仅使用 source 标注训练轻量旋转 RPN/ROI 检测头。BrightAug 作为通用检测器保留，DINO 分支只在预定义的低照度困难范围内启用，二者不进行分数融合。

### 暗光专项结果

> 在 `real_seq02[137..169]` 暗光大目标困难切片上，BrightAug 的 top-1 命中率为 `0/33`，冻结 DINOv2 分支将其提升至 `29/33`，旋转 IoU 连续最大失败长度由 `33` 降低到 `1`。该结果说明冻结 DINOv2 能够恢复暗光条件下的语义目标排序。

### 完整 test 结果

> 在 992 帧完整测试集上，稳定化 ScopedDINO 将全帧 real 中心召回率从 `0.6476` 提升至 `0.7238`，Weighted center score 从 `0.8943` 提升至 `0.9171`，mAP 从 `0.81683` 提升至 `0.81750`。同时 real DFR 从未稳定化 DINO 的 `3.8323%/frame` 降至 `2.6402%/frame`，ACI 从 `0.9182` 提升至 `0.9395`；sim 域全部指标保持不变。

### 限制

> 由于低照度 scope 来自 target-dev 困难切片，该结果用于验证方法机制和暗光专项有效性，而不是完全独立的跨序列 zero-shot 泛化声明。完整 real 域的剩余 MCML 主要由远距离候选不足和小目标几何误差造成，属于后续独立研究问题。

## 11. 可复现实验命令

### DINO 小头训练

```bash
PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_rotated_labeller.py \
  --data-root crane_project/data/crane_grab/ \
  --source-train-datasets train:train train_sim:train \
  --source-val-datasets val:val \
  --dinov2-repo third_party/dinov2 \
  --dinov2-checkpoint pretrained/dinov2_vitl14_pretrain.pth \
  --dinov2-model dinov2_vitl14 \
  --dino-gpus 1 2 --head-gpu 0 \
  --legacy-sdpa-query-chunk 512 \
  --dino-height 600 --dino-max-long-side 1333 --patch-size 14 \
  --rpn-feat-channels 256 --roi-fc-channels 1024 \
  --roi-samples 256 --proposal-count 2000 --max-detections 2000 \
  --epochs 8 --lr 0.001 --momentum 0.9 --weight-decay 0.0001 \
  --max-grad-norm 10 --warmup-iters 1000 --warmup-ratio 0.001 \
  --lr-steps 5 7 --lr-gamma 0.1 \
  --checkpoint-interval 1 --selection-epochs 1 2 3 4 5 6 7 8 \
  --feature-cache-dir work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache \
  --work-dir work_dirs/dino_teacher_scoped_lowlight_v1_formal8 \
  --seed 0 \
  --out-json work_dirs/dino_teacher_scoped_lowlight_v1_formal8/train_result.json \
  --skip-target-eval
```

target-dev 暗光诊断结果作为方法形成过程的历史证据保留在文档和结果文件中；
正式组合推理与完整 test 统一使用 14.4 节的标准 `tools/test.py` 命令，避免维护
另一套离线结果拼接脚本。

主要输出：

```text
work_dirs/dino_teacher_scoped_lowlight_v1_formal8/train_result.json
work_dirs/dino_teacher_scoped_lowlight_v1_formal8/labeller_best_source_only.pth
work_dirs/dino_teacher_scoped_lowlight_v1_formal8/formal_integrated_test/results.pkl
```

## 12. 当前阶段结论

当前可以冻结的论文创新点是：

> 在 BrightAug 检测器之外，引入冻结 DINOv2 语义表征和 source-only 旋转检测头，用于恢复暗光大目标中的目标语义排序；该分支不破坏 sim 域能力，并在 target-dev 暗光大目标切片上显著降低连续漏检。

当前还不能宣称：

- 已经解决完整 real test 的所有 MCML；
- 已经解决远距离/小目标候选生成；
- 已经完成不依赖 target-derived scope 的最终部署方法；

后续小目标研究应在这份创新点记录冻结后单独建立新的实验分支，不要把 `real_seq02[2..41]` 或 `real_seq03[129..192]` 的几何问题回写成暗光 DINO 的失败。

## 13. 暗光恢复框的因果稳定化验证

只读框稳定性审计确认，ScopedDINO 的 DFR 退化主要来自其在
`real_seq02[137..169]` 新恢复的连续输出，而不是 scope 外结果被改写：

```text
real_seq02 output-stream DFR: 2.1275 -> 4.5157
共同有输出转移 DFR:          2.1275 -> 2.2816
新增可观测转移:              31
scope 外变化帧:              0
```

因此新增一个不训练模型的
`Source-Selected Causal DINO Box Stabilization Probe`。它只对 scope 内
top-1 DINO 框的宽、高和角度进行因果 EMA，不改变：

- 检测是否存在；
- 候选顺序和分数；
- 框中心；
- scope 外任何检测结果。

旋转框平滑先消除 `(w,h,theta)` 与 `(h,w,theta+pi/2)` 的等价表示歧义，
再在 log-width/log-height 和 `sin(2theta), cos(2theta)` 空间插值。平滑系数
只允许从 `{0.25, 0.5, 0.75, 1.0}` 中选择，其中 `1.0` 是严格不修改输出的
保底项。

系数选择只使用正式 source validation，必须同时满足：

- source-val top-1 命中数不下降；
- source-val MCML 不增加；
- source-val mean RIoU 下降不超过 `0.005`；
- source-val ACI 不下降。

满足约束后选择 source-val DFR 最低的系数，再将固定系数一次性用于完整
test 结果。target 标签只用于事后诊断，不能参与系数选择。完整 test 已同时
保持暗光 top-1/MCML 并改善 DFR/ACI，因此固定 `alpha=0.25` 已并入最终方法。

## 14. 正式模型化整合

### 14.1 最终顺序推理流程

最终实现不再依赖先生成多个 pickle 再用离线工具拼接。正式 config 直接构建
`ScopedDinoLowlightDetector`：

```text
当前帧
  -> BrightAug 主检测器
  -> 读取预定义低照度 scope
       scope 关闭：原样返回 BrightAug，并清空时序状态
       scope 开启：
         -> 冻结 DINOv2 ViT-L/14
         -> source-only Oriented RPN/ROI head
         -> valid-content 过滤
         -> DINO top-1；无 DINO 输出时回退 BrightAug
         -> alpha=0.25 因果框稳定器
              中心、分数、排序、输出存在性不变
              只平滑 log(w)、log(h) 和 pi 周期角度
  -> MMRotate 标准一类别检测结果
```

时序状态在以下情况清空：scope 关闭、检测 silence、序列变化或帧号不连续。
因此不会跨序列、跨困难片段传播历史框。

### 14.2 正式代码位置

```text
组合 detector:
  mmrotate/models/detectors/scoped_dino_lowlight_detector.py

唯一正式 config:
  crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py

完整论文指标入口:
  mmrotate/datasets/crane_custom_dota.py
  crane_project/tools/eval_crane_offline.py
```

`CraneDataset.evaluate()` 由 config 的 `paper_temporal=True` 启用完整指标，
一次标准 `tools/test.py` 会同时输出标准 mAP、全帧中心召回、
`R_center^det@15px`、`R_center^all@25px`、DFR、ACI、TDR、MCML、MRF 和 mean RIoU。

旧的 audit/tool 文件仍保留为论文证据和回归检查，但不再承担最终推理组合。

### 14.3 三张 8 GB GPU 的部署方式

```text
GPU 0：BrightAug + 冻结 rotated RPN/ROI head
GPU 1：DINOv2 前 12 个 transformer blocks
GPU 2：DINOv2 后 12 个 transformer blocks
```

BrightAug 和 DINO head 按帧顺序执行，不同时保留两套前向激活；进入 DINO
前主动释放 BrightAug CUDA cache。该模型只支持非分布式、batch size 1 的
顺序推理，因为因果稳定器需要严格的帧序和单一状态流。

### 14.4 正式完整 test 命令

```bash
PYTHONPATH=. python3 tools/test.py \
  crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py \
  work_dirs/crane_symeood_k1_brightaug/epoch_20.pth \
  --gpu-ids 0 \
  --out work_dirs/dino_teacher_scoped_lowlight_v1_formal8/formal_integrated_test/results.pkl \
  --eval mAP \
  --work-dir work_dirs/dino_teacher_scoped_lowlight_v1_formal8/formal_integrated_test
```

命令行 checkpoint 仍是未经修改的 BrightAug checkpoint；DINO 小头 checkpoint、
DINOv2 权重、scope 和稳定器系数均由 config 固定加载。

### 14.5 数据边界

最终训练权重没有使用 test 图像或标签：DINOv2 冻结，旋转检测头只在
`train + train_sim` 上训练并由 official `val` 选择，稳定器系数也只由 official
`val` 选择。但当前 scope manifest 明确标记为 `target_label_derived=true`，所以
当前完整 test 仍是 target-dev 机制诊断，不可声称是完全独立的 zero-shot final
test。代码已把这一限制固化在 manifest 中，没有把 target 标签写入权重。

## 15. 最终论文创新点定义

> 针对暗光大目标场景中几何候选仍存在但分类排序坍塌的问题，本文提出一种
> scope-gated frozen-DINO rotated rescue detector。该方法冻结 DINOv2 ViT-L/14，
> 仅使用 source 标注训练旋转 RPN/ROI 小头，并在低照度模式内以 DINO 语义
> 路径替代失效的 anchor/FPN 分类路由；随后采用完全由 source validation 选择的
> 因果旋转框稳定器，在不改变检测中心、分数、排序和输出存在性的前提下平滑
> 尺度与周期角度。完整顺序推理中 scope 外严格保留 BrightAug，因而实现暗光
> 大目标排序恢复、时序稳定性恢复和非目标域能力隔离。
