# 冻结 DINOv2 暗光大目标语义救援：完整技术记录

更新时间：2026-07-26

当前状态：`formal8` source-only 正式训练正在运行，最终 best epoch 和 target-dev 结果尚未产生。

## 1. 问题定义

本路线只解决：

```text
test/real_seq02[137..169]
```

该片段是暗光大目标分类排序失败，不是一般意义上的“所有特征都变弱”。现有诊断已经确认：

- 31/33 帧仍存在 `RIoU >= 0.5` 的可用候选；
- 帧 164、167 是独立 geometry miss；
- 可用候选历史排名中位数约 5708；
- source 稳定监督路由集中于 `FPN level1 + anchor0`；
- target 正确候选集中于 `FPN level0 + anchor2`；
- target 可用 FPN 特征能量不弱，但传统分类器方向失配并把正确候选压到深层。

必须与下面的远距小目标问题分开：

```text
test/real_seq02[2..41]
```

后者主要是分辨率和候选几何不足，当前 DINO 暗光语义救援不负责解决。

## 2. 创新点概述

方法暂定名：

```text
Scope-Gated Frozen DINO Semantic Rescue
范围受控的冻结 DINO 语义救援
```

核心思想：

1. 保留 BrightAug 作为通用检测器，不修改其 backbone、FPN、分类和回归路径。
2. 引入冻结的 DINOv2 ViT-L/14，使用其更稳定的 patch 语义表示。
3. 在冻结 DINO 特征上独立训练 oriented RPN 和 rotated ROI 检测小头。
4. DINO 小头只使用 source 标注训练，并独立使用 source validation 选权。
5. 在预先定义的暗光工作范围内由 DINO 专家输出；范围外严格保留 BrightAug。
6. 不进行 BrightAug/DINO score 数值融合，不比较两个模型未经标定的绝对分数。

该设计借鉴 CVPR 2025 DINO Teacher 的“冻结大规模 DINO 表征 + source-only labeller”思想，但不是对原论文完整 DAOD 流程的复现：

- 当前实现包含 frozen-DINO labeller；
- 当前没有 target pseudo-label student training；
- 当前没有 DINO feature alignment student branch；
- 因此论文中应称为针对旋转检测的受控改造，而不是完整复现 DINO Teacher。

## 3. 模型结构

### 3.1 BrightAug 通用分支

- 配置：`crane_project/configs/crane_symeood_k1_brightaug.py`
- 作用：正常光照、远距和非暗光范围继续使用原检测能力。
- checkpoint：由 `crane_project/tools/ckpt_sweep.py` 在官方 val 上选择。
- 候选 epoch：`16, 18, 20, 22, 24`。
- 正式组合运行时读取：

```text
work_dirs/crane_symeood_k1_brightaug/ckpt_sweep/selected_checkpoint.txt
```

BrightAug epoch 与 DINO 小头 epoch 没有对应关系。

### 3.2 冻结 DINOv2 语义分支

- 模型：DINOv2 ViT-L/14；
- patch size：14；
- 输入高度：600；
- 最大长边：1333；
- 输出：最后一层 patch token 还原成单尺度空间特征图；
- GPU 1、2：分片放置冻结 DINO blocks；
- GPU 0：训练 RPN/ROI 小头；
- Python 3.8/旧 PyTorch：legacy SDPA，query chunk=512；
- DINO 全部参数 `requires_grad=False`，运行结束检查参数版本未变化。

### 3.3 Oriented RPN

- 类型：`OrientedRPNHead`；
- 单尺度 stride：14；
- anchor size：32、64、128、256、512；
- anchor ratio：0.5、1.0、2.0；
- proposal count：2000；
- RPN feature channels：256。

### 3.4 Rotated ROI

- 类型：`OrientedStandardRoIHead`；
- ROIAlign：rotated 7x7；
- bbox head：`RotatedShared2FCBBoxHead`；
- ROI FC channels：1024；
- ROI samples：256；
- class count：1；
- bbox regression：class-agnostic rotated OBB regression；
- max detections：2000。

只有 oriented RPN 和 rotated ROI head 参与优化。

## 4. 已验证的旧版 source-only 结果

旧实验使用：

```text
val/real_seq07：226 帧
frame_id % 5 != 0：181 帧 source-train
frame_id % 5 == 0：45 帧 source-val
```

重要结论：

- 没有使用 `test/real_seq02` 训练；
- seq02 标签只在 source checkpoint 固定后用于诊断；
- 历史 source-val：45/45 top-1；
- 历史 target-dev：`real_seq02[137..169]` 达到 top1=32/33、MCML=1；
- 该结果证明冻结 DINO 语义分支有能力绕开 BrightAug 的 anchor/FPN 分类路由失配。

限制：181 帧来自 `val/real_seq07`，不是官方 train/val 协议，只能作为 legacy 诊断证据。

## 5. 当前 formal8 正式训练协议

配置：

```text
crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py
```

统一入口：

```text
crane_project/tools/dino_teacher_scoped_method.py
```

训练实现：

```text
crane_project/tools/dino_teacher_rotated_labeller.py
```

### 5.1 数据

```text
source train：train + train_sim，共 2781 帧
source val：完整官方 val，共 738 帧
target-dev：test/real_seq02[137..169]，训练完成后才读取
```

训练阶段固定使用：

```text
--skip-target-eval
```

因此训练 JSON 应满足：

```text
target_used_for_training = False
target_used_for_checkpoint_selection = False
target_labels_first_used = not_read
```

### 5.2 DINO 小头独立训练参数

```text
max epochs             = 8
optimizer              = SGD
learning rate          = 0.001
momentum               = 0.9
weight decay           = 0.0001
gradient clip          = max_norm 10
warmup                 = linear
warmup iterations      = 1000
warmup ratio           = 0.001
LR step epochs         = 5, 7
LR gamma               = 0.1
checkpoint interval    = 1 epoch
selection epochs       = 1,2,3,4,5,6,7,8
seed                    = 0
```

2781 张训练图下，8 epoch 约为 22,248 次一图一更新的 head optimizer step。该预算属于 DINO 小头，不与 BrightAug 24 epoch 机械对应。

### 5.3 DINO checkpoint 选择

每个 epoch 都保存和执行 source-val。选择顺序：

1. `source-val top1_hits` 更高；
2. `R@20` 更高；
3. `R@100` 更高；
4. `mean top1 RIoU` 更高；
5. 完全相同时保留更早 epoch。

最终推理使用：

```text
work_dirs/dino_teacher_scoped_lowlight_v1_formal8/
labeller_best_source_only.pth
```

不是固定使用 epoch8。

### 5.4 当前运行状态

截至当前日志：

- epoch1 已运行至少 625/2781；
- 最近 25 样本 loss 从约 1.06 平稳下降到约 0.07；
- 未出现 NaN 或梯度爆炸；
- `cache=0/N` 表示第一次没有命中旧 DINO 特征，并不表示没有写缓存；
- 每张图的 FP16 DINO 特征在首次前向后立即写入磁盘；
- epoch2 应逐渐显示 `cache=N/N`，并明显加速；
- 正式 best epoch、source-val 和 target-dev 结果仍待训练结束。

工作目录：

```text
work_dirs/dino_teacher_scoped_lowlight_v1_formal8
```

## 6. 推理组合

当前实现的核心策略：

```text
scope enabled  + DINO有输出 -> DINO top-1
scope enabled  + DINO无输出 -> BrightAug fallback
scope disabled             -> 原样返回 BrightAug
```

设计原则：

- BrightAug 与 DINO 的绝对 score 不可直接比较；
- 不做加权 score fusion；
- 不允许 DINO 在全部场景无条件覆盖 BrightAug；
- real_seq03/sim_seq09 的旧审计已经证明全局覆盖会产生负迁移；
- 当前 scope manifest 来自 target-dev 协议，因此只能称为 diagnosis-only。

## 7. 保留的两个 val 训练方案

等 formal8 结果出来后，再决定是否执行。两者当前都是“计划”，不是已完成实验。

### 方案 A：Legacy reproduction

目的：确认之前 32/33、MCML=1 能否稳定复现。

```text
训练：val/real_seq07 中 frame_id % 5 != 0，共181帧
验证：val/real_seq07 中 frame_id % 5 == 0，共45帧
work_dir：必须独立
```

定位：诊断复现，不作为最严格的官方 train/val 主结果。

### 方案 B：TrainVal fixed-step refit

目的：在不使用 target 选权的前提下利用全部 source 标注。

步骤：

1. 先完成当前 formal8；
2. 只根据 formal8 的 source-val 固定 best epoch 或 optimizer update 数；
3. 从头初始化新的 DINO RPN/ROI 小头；
4. 训练数据改成 `train + train_sim + val`；
5. 严格训练到预先固定的 update 数；
6. 不再使用 val 做 checkpoint 选择；
7. 训练完成后才评估 target-dev。

当前代码会阻止同一份 val 同时用于训练和 checkpoint 选择，这是正确保护。执行方案 B 前需要新增专门的 fixed-step refit stage。

## 8. formal8 结果出来后的决策

需要同时读取：

```text
source best_epoch
source-val top1/R@20/R@100/mean RIoU
real_seq02[137..169] top1/MCML/mean RIoU
real_seq03/sim_seq09 非回退结果
```

建议裁决：

- formal8 接近或达到旧 32/33：保留 formal8 为主模型，不重新打开 val 训练；
- formal8 source-val 良好但 target 明显下降：执行方案 A 复现和方案 B fixed-step refit；
- formal8 source-val 本身不稳定：优先检查小头训练、数据域比例和 checkpoint 选择，不允许直接根据 seq02 选择 epoch；
- 方案 A 恢复而 formal8/方案 B 不恢复：论文应把 real_seq07 source coverage 作为关键因素，并缩小泛化声明；
- 方案 B 恢复且 source/holdout 不退化：可作为最终 source-only 模型候选。

## 9. 论文报告边界

必须分开报告：

1. source-val checkpoint selection；
2. `real_seq02[137..169]` 暗光大目标困难子集；
3. 完整 real_seq02；
4. `real_seq02[2..41]` 远距几何失败；
5. real_seq03/sim_seq09 范围外非回退。

`real_seq02[137..169]` 已参与问题分析、方法设计和 scope 设计，因此应称为 target-dev 或 low-light hard subset，不能声称是完全未观察过的 zero-shot test。

这不否定创新有效性，但限制“跨未知暗光序列泛化”的声明强度。

## 10. 关键输出和命令

训练：

```bash
PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_scoped_method.py \
  --config crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py \
  --stage train
```

target-dev 诊断：

```bash
PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_scoped_method.py \
  --config crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py \
  --stage test
```

训练结果：

```text
work_dirs/dino_teacher_scoped_lowlight_v1_formal8/train_result.json
```

target-dev 结果：

```text
work_dirs/dino_teacher_scoped_lowlight_v1_formal8/target_dev_test_result.json
```

## 11. 当前代码检查状态

最后一次本地检查：

```text
Python compile passed
DINO-related tests: 112 passed
git diff --check passed
```

本地没有执行 CUDA 训练；服务器 formal8 运行结果是下一项决定性证据。

