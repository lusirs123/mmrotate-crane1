# 路径 A 可信性分析与实现细节

> 路径 A = 推理期条件门控（`global_max < score_thr`）+ 强调制组合
> 核心假设：strong 调制帮了 hard-slice 但伤了 easy frames，门控保留帮助、去掉伤害

---

## 一、执行摘要

**结论：路径 A 当前可信度为低到中，存在一个未验证的核心假设和一个代码级测量缺陷。在投入训练实验前，必须先修复 probe 并获取三个可比数据点。**

三个关键发现改变了判断基础：

1. **probe 测错了东西**：`ctx_entry_probe.py` 调用 `model.extract_feat()` 取 FPN 特征，不经过 injector 调制。对 injector 模型，probe 测的是「训练有 injector、推理不 inject」的 train-test mismatch 场景，而非实际推理行为。0.143 是 mismatch 数值，不是真实推理数值。

2. **0.691 vs 0.143 的比较完全无效**：前者来自 `crane_eood_k1`（EOOD 模型）的 aux1 头（RotatedATSSHead），后者来自 `crane_symeood_k1_platform_injector`（SymEOOD 模型）的 main 头（SymEOODHead）。不同模型、不同检测头、不同注入状态——三层不可比。

3. **分数 gap 极大**：死段 `global_max ≈ 0.006–0.01`，需提升到 `score_thr=0.05` 才能出框，gap 近 8–10 倍。feature 层调制能否间接产生这么大的 scoring 变化，存疑。

---

## 二、代码审查发现

### 2.1 当前推理流程（sym_eood_detector.py 第 249–263 行）

```python
def simple_test(self, img, img_metas, rescale=False):
    feat = self.extract_feat(img)           # backbone → neck → FPN 特征
    if self.platform_context_injector is not None:
        feat = self.platform_context_injector.forward_test_features(feat)  # 无条件调制
    results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
    ...
```

**关键：injector 对所有帧无条件调制，没有门控。**

### 2.2 调制公式（platform_context_injector.py 第 37–58 行）

```python
def _modulate_with_preds(self, feats, preds, apply=True):
    ...
    gate = logit.tanh()                                    # 平台 context map → tanh
    scale = self.gate_scale * torch.tanh(self.gate_alpha)  # 可学习标量
    out_feats[level] = out_feats[level] * (1.0 + scale * gate)  # 乘性调制
```

| 配置 | gate_scale | init_gate_alpha | scale 起始值 | 行为 |
|---|---|---|---|---|
| ordinary | 0.15 | 0.0 | 0.0 | 训练初期不调制，靠 alpha 逐步移动 |
| strong | 0.30 | 0.50 | 0.139 | 从第一轮就调制，且强度大 |

**关键：scale 是全局标量（非逐帧），gate 是逐像素的。modulation = `feat * (1 + scale * gate)`。**

当 scale=0 时，modulation 是恒等（`feat * 1.0`）——这是 ordinary 的初始状态。
当 scale=0.14 时，feat 最多被放缩到 ±14%——这是 strong 的初始状态，已经不是 near-identity。

### 2.3 probe 的测量缺陷（ctx_entry_probe.py 第 372–377 行）

```python
with torch.no_grad():
    feat = model.extract_feat(img_tensor)      # ← 只取 backbone+neck
    candidate_head, cls_scores, bbox_preds = forward_candidate_head(
        model, feat, args.candidate_source)    # ← 用未调制特征
```

`extract_feat` 只走 `backbone → neck`，**不经过 injector**。所以：

- 对 baseline 模型（无 injector）：probe 测的就是推理行为 ✓
- 对 injector 模型：probe 测的是 **不 inject 的 mismatch 场景** ✗

实际推理（`simple_test`）是 `extract_feat → inject → bbox_head`，有调制。但 probe 从未测过这个路径。

**这意味着：我们不知道 injector 在实际推理时对 hard-slice 几何的影响。0.143 不是推理数值。**

### 2.4 0.691 vs 0.143 比较的三层不可比

| 维度 | 0.691（设计文档 §12 Gate-0b） | 0.143（用户分享 Stage2 ordinary） |
|---|---|---|
| 模型 | crane_eood_k1（EOOD 基线） | crane_symeood_k1_platform_injector（SymEOOD+injector） |
| 检测头 | aux1 = RotatedATSSHead | main = SymEOODHead |
| 注入状态 | 无 injector | probe 不 inject（mismatch） |
| 帧范围 | 133–171（39帧） | 137–169（33帧） |

EOOD 和 SymEOOD 是不同模型：不同 loss（Focal+L1+IoU vs SymNFL+SymKLD）、不同 assigner（MaxIoU+Pola vs SymPOLA）、不同训练动态。

aux1（ATSS）和 main（SymEOODHead）是不同头：不同 anchor 设置（ATSS ratios=[1.0] vs SymEOODHead ratios=[0.5,1.0,2.0]）、不同标签分配。

subthreshold probe 数据也印证了头差异：`aux1/dead mean_best_roi_max=0.0256` vs `main/dead mean_best_roi_max=0.0025`——aux1 的亚阈值峰比 main 强 10 倍。

### 2.5 baseline 三个死段

`failure_segments.json` 显示 symeood_k1 baseline 在 real_seq02 有三个死段：

| 死段 | 长度 | gt_diag_mean | 类型 | 特征 |
|---|---|---|---|---|
| [129..172] | 44帧 | 156 | 近处大目标 | MCML_max=44，主死段 |
| [2..41] | 40帧 | 90 | 远处小目标 | 第二大死段 |
| [111..127] | 17帧 | 154 | 近处大目标 | 第三死段 |

**关键：[129..172] 是近处大目标（gt_diag≈156px），不是「太小看不见」。这印证了 DEAD-global 是外观/打分问题，不是尺寸问题。**

Path A 的门控 `global_max < 0.05` 会同时触发这三个死段。但 [2..41] 是远处小目标——平台上下文对这种场景的帮助可能不同（平台也小、信号也弱）。

### 2.6 时序重锚揭示的双重困难

`temporal_reanchor_probe`（aux1 head, oracle-gt-start, seq02[129..172]）数据：

| Frame | reanchor_riou | center_dist | reanchor_score | brightness |
|---|---|---|---|---|
| 129 | 0.509 ✓ | 14px | 0.048 | 73 |
| 130 | 0.392 | 32px | 0.045 | 77 |
| 131 | 0.183 | 46px | 0.044 | 82 |
| 132 | 0.132 | 61px | 0.074 | 83 |
| 140 | 0.000 | 208px | 0.014 | 52 |
| 141 | 0.000 | 217px | 9e-6 | 49 |
| 150 | 0.000 | 382px | 2e-9 | 46 |
| 172 | 0.000 | 750px | 2e-9 | 102 |

**双重困难：**

1. **分数崩塌**：Frame 129 勉强命中（RIoU=0.509, score=0.048），但 Frame 130 就差一点（RIoU=0.392）。Frame 141+ 预测位置分数降到 1e-6 量级。

2. **运动漂移**：匀速预测从 Frame 131 开始失效。center_dist 从 14px（F129）增长到 750px（F172）。抓斗在死段内不是匀速运动——可能是加速/减速/变向。

3. **亮度无关性**：brightness 从 73（F129）降到 46（F150）再升到 102（F172），但全程都是死段。这与设计文档结论一致：「global_max 与亮度无关 → 是顶梁外观对 SymEOOD 打分整体 OOD」。

**对路径 A 的启示**：门控信号 `global_max < 0.05` 能识别死段，但调制后的特征需要跨越近 8 倍的分数 gap（0.006→0.05）。同时，死段内抓斗位置在快速变化，调制只在特征层作用——不解决位置预测问题。

---

## 三、路径 A 可信性分析

### 3.1 核心假设的验证状态

| 假设 | 验证状态 | 证据 |
|---|---|---|
| strong 调制改善了 hard-slice 几何 | **未验证** | probe 从未测过 with-injection；0.143 是 mismatch 数值 |
| strong 调制伤害了 easy frames | **部分验证** | MCML_max 从 44(baseline) → 47(ordinary) → 71(strong)，但不确定是原死段恶化还是新死段出现 |
| 门控能分离 help/hurt | **理论成立** | global_max 在死段≈0.006-0.01，健康帧>>0.05，分离干净 |
| L_ctx-consist 能消除 mismatch | **未验证** | 原设计 §5 提出但从未实现 |

### 3.2 结构性问题：作用层错位

DEAD-global 的诊断结论是「**分数病非几何病**」——GT 邻域 decoded 框 RIoU 0.58–0.84（eood_k1/aux1）一直存在，只是 cls/objectness 分数排不上来。

路径 A 的门控 + 调制仍然在 **feature 层**做文章：

```
extract_feat → [gate decision] → inject (modulate FPN features) → bbox_head (cls + reg) → output
                                              ↑
                                    作用层：feature
                                    问题层：scoring (cls head 对 OOD 外观的响应)
```

feature 调制改变的是输入 bbox_head 的特征。bbox_head 的 cls 分支对这些特征的响应如果本身就是 OOD 的（对死段外观打极低分），改变特征能否让 cls 分支改变打分？

这是一个间接路径：`feature change → cls head response change → score change`。间接路径的信号会被稀释。

**ordinary injector 的证据**：decoded-neighborhood 在 Stage1→Stage2 的趋势是 0.037→0.143（4 倍），但这是 mismatch 数值。即使真实推理数值翻倍到 0.3，离 usable@0.50 还有距离。而且这是几何指标，不是分数指标——几何好了不代表分数能过阈值。

### 3.3 分数 gap 问题

死段 `global_max ≈ 0.006–0.01`，需要提升到 `score_thr=0.05`。

| 指标 | 死段当前值 | 需要达到 | gap |
|---|---|---|---|
| global_max | 0.006–0.01 | 0.05 | 5–8 倍 |
| decoded-neighborhood (mismatch) | 0.143 | 0.50 | 3.5 倍 |

feature 调制需要在 cls 分支上间接产生 5–8 倍的分数提升。这是一个很大的 ask。

对比：ordinary injector 把 decoded-neighborhood 从 Stage1 的 0.037 提到 0.143（4 倍，但仍是 mismatch）。如果 with-injection 的提升幅度类似，decoded-neighborhood 可能到 0.3-0.5——接近但不确定能过 0.50。而分数的提升幅度可能更小（间接路径）。

### 3.4 Train-test mismatch 分析

| 训练策略 | 推理策略 | mismatch 严重度 | 需要的额外组件 |
|---|---|---|---|
| always-on (ordinary, scale≈0) | gate off/on | **低**（scale≈0 时 near-identity） | 无 |
| always-on (strong, scale≈0.14) | gate off/on | **高**（scale=0.14 不是 near-identity） | L_ctx-consist |
| conditional (match inference) | gate off/on | **无** | 训练期也需 gate 逻辑 |

**如果用 ordinary 强度 + 门控**：mismatch 低，但 ordinary 可能不够强（decoded-neighborhood 仅 0.143，gap 大）。

**如果用 strong 强度 + 门控**：mismatch 高，需要 L_ctx-consist。但 L_ctx-consist 从未实现，效果未知。且 L_ctx-consist 可能把调制约束得太弱（在 easy frames 上 near-identity），变成 ordinary。

这是一个两难：ordinary 不够强，strong 需要 L_ctx-consist 但 L_ctx-consist 可能把 strong 变成 ordinary。

### 3.5 门控信号的可获得性

门控信号是 `global_max`（unmodulated main head 的 cls score map 最大值）。

获取方式：在 `simple_test` 中，先跑一次 `bbox_head(feat)` 取 cls_scores，算 global_max，再决定是否 inject。

```python
# 伪代码
feat = self.extract_feat(img)
if self.platform_context_injector is not None and self.use_gate:
    outs = self.bbox_head(feat)  # 额外一次 forward
    cls_scores = outs[0] if isinstance(outs, tuple) else outs
    global_max = max(cs.sigmoid().max().item() for cs in cls_scores)
    if global_max < self.gate_threshold:
        feat = self.platform_context_injector.forward_test_features(feat)
results = self.bbox_head.simple_test(feat, ...)
```

**代价**：hard frames 多一次 bbox_head forward。但 hard frames 稀有（~100/2600 ≈ 4%），平均开销 <2%。

**问题**：`bbox_head(feat)` 和 `bbox_head.simple_test(feat)` 之间有冗余 forward。可以优化为在 simple_test 内部提取 global_max，但需要改 bbox_head 代码，更侵入。

### 3.6 baseline 死段 vs injector 死段的对比

| 指标 | baseline (symeood_k1) | ordinary injector | strong injector |
|---|---|---|---|
| MCML_max | 44 (seq02[129..172]) | 47 | 71 |
| TDR_w10 | ? | 71.14 | 59.45 |
| sim A-RMSE | 1.3467 | 1.3479 | 1.341 |
| real R_center | ? | 99.55 | 99.36 |

MCML_max 从 44→47→71 单调恶化。这可以是：
- **解释 A**：调制在原死段无帮助，且在健康帧引入新死段 → 门控无用
- **解释 B**：调制在原死段有帮助（缩短了），但在其他地方引入了更长的新死段 → 门控可能有用

**没有 per-sequence MCML 分析，无法区分 A 和 B。**

如果 strong injector 的 MCML_max=71 来自原死段恶化（44→71），路径 A 死。
如果来自新死段（原死段缩短到 <44，但新地方出现了 71 帧死段），路径 A 可能有救。

---

## 四、实现细节

### 4.1 推理期门控（simple_test 改造）

```python
def simple_test(self, img, img_metas, rescale=False):
    feat = self.extract_feat(img)

    if self.platform_context_injector is not None:
        if getattr(self, 'use_context_gate', False):
            # 门控模式：先取 unmodulated global_max
            with torch.no_grad():
                outs = self.bbox_head(feat)
                cls_scores = outs[0] if isinstance(outs, tuple) else outs
                global_max = max(
                    cs.sigmoid().max().item() for cs in cls_scores)
            if global_max < self.gate_threshold:
                feat = self.platform_context_injector.forward_test_features(feat)
        else:
            # 无门控模式（当前行为）：always inject
            feat = self.platform_context_injector.forward_test_features(feat)

    results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
    bbox_results = [
        rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        for det_bboxes, det_labels in results_list
    ]
    return bbox_results
```

新增配置项：
- `use_context_gate`: bool，是否开启门控
- `gate_threshold`: float，默认 0.05（复用 score_thr）

### 4.2 L_ctx-consist（训练期一致性损失）

目的：让调制在 easy frames 上近恒等，使推理期 gate-off 不产生 mismatch。

```python
# forward_train 改造（伪代码）
x = self.extract_feat(img)
x_unmod = x  # 保存未调制特征

if self.platform_context_injector is not None:
    x_mod, injector_losses = (
        self.platform_context_injector.forward_train_features(
            x, img_metas, gt_bboxes))
else:
    x_mod = x

main_outs = self.bbox_head(x_mod)  # 调制路径

# L_ctx-consist: easy frames 上 modulated ≈ unmodulated
if self.platform_context_injector is not None and self.use_ctx_consist:
    unmod_outs = self.bbox_head(x_unmod)  # 未调制路径
    # 逐帧计算 global_max 作为 easy/hard 判据
    for i in range(len(img_metas)):
        global_max_i = max(
            cs[i].sigmoid().max().item() for cs in unmod_outs[0])
        if global_max_i > self.consist_threshold:
            # easy frame: 约束 modulated ≈ unmodulated
            for lvl in range(len(main_outs[0])):
                loss_cls_i = F.mse_loss(
                    main_outs[0][lvl][i], unmod_outs[0][lvl][i])
                loss_reg_i = F.mse_loss(
                    main_outs[1][lvl][i], unmod_outs[1][lvl][i])
                losses['loss_ctx_consist'] = (
                    losses.get('loss_ctx_consist', 0)
                    + (loss_cls_i + loss_reg_i) * self.consist_weight)
```

新增配置项：
- `use_ctx_consist`: bool
- `consist_threshold`: float，默认 0.05（与 gate_threshold 一致）
- `consist_weight`: float，默认 0.1

**代价**：训练时 easy frames 多一次 bbox_head forward。但 easy frames 是多数，所以大部分 batch 都有额外开销。可以考虑只在部分 iteration 上算 consist loss（如每 4 步算一次）。

### 4.3 训练策略选择

| 方案 | 训练 | 推理 | 优点 | 缺点 |
|---|---|---|---|---|
| A1: ordinary + gate | always-on, scale≈0 | gate off/on | mismatch 低，无需 L_ctx-consist | ordinary 可能不够强 |
| A2: strong + gate + L_ctx-consist | always-on + consist | gate off/on | 可以用强调制 | L_ctx-consist 可能把 strong 变成 ordinary |
| A3: strong + conditional train | train 也 gate | gate off/on | 无 mismatch | hard frames 少，调制欠拟合 |

**推荐 A1 先行**：最简单、最低风险。如果 A1 显示门控本身不伤 easy frames（因为 ordinary scale≈0 近恒等），且 hard frames 有改善，再上 A2 试 strong。

### 4.4 probe 修复

当前 `ctx_entry_probe.py` 不 inject。需要加一个 `--apply-injection` flag：

```python
# analyze_frame 内，extract_feat 之后：
feat = model.extract_feat(img_tensor)
if args.apply_injection and hasattr(model, 'platform_context_injector') \
        and model.platform_context_injector is not None:
    feat = model.platform_context_injector.forward_test_features(feat)
candidate_head, cls_scores, bbox_preds = forward_candidate_head(
    model, feat, args.candidate_source)
```

---

## 五、前置诊断（必须先做）

### 5.1 三个 probe（用现有 checkpoint，零训练）

| Probe | 模型 | 注入 | 命令核心 |
|---|---|---|---|
| P1 | symeood_k1 baseline (epoch_24) | 无 injector | `--config crane_symeood_k1.py --checkpoint .../epoch_24.pth` |
| P2 | ordinary injector (epoch_24) | 不 inject（当前行为） | `--config crane_symeood_k1_platform_injector.py --checkpoint .../epoch_24.pth` |
| P3 | ordinary injector (epoch_24) | **with injection**（修 probe 后） | 同上 + `--apply-injection` |

帧范围统一：`--seq real_seq02 --start 137 --end 169 --candidate-source main`

### 5.2 判据

| 场景 | P1 (baseline) | P2 (injector, no inject) | P3 (injector, with inject) | 判断 |
|---|---|---|---|---|
| α | 0.3 | 0.143 | 0.5+ | injector 帮了 → **Path A 值得做** |
| β | 0.3 | 0.143 | 0.3 | injector 没帮 → **Path A 死** |
| γ | 0.5+ | 0.143 | 0.3 | injector 伤了 → **方向错误，Path A 死** |
| δ | 0.143 | 0.143 | 0.5+ | injector 训练改变了 backbone，inject 才有效 → **Path A 值得做** |

**关键比较是 P2 vs P3**（同一个模型，注入 vs 不注入），这直接回答「injection 是否改变 hard-slice 几何」。

P1 提供 baseline 参照——如果 P1≈P2，说明 injector 训练没改变 backbone 的 unmodulated 表现；如果 P1≠P2，说明 injector 训练改变了 backbone 权重。

### 5.3 额外诊断（per-sequence MCML）

对 ordinary 和 strong injector 的 test 结果做 per-sequence MCML 分析：
- seq02[129..172] 是否缩短？
- 新死段出现在哪里？

这区分 3.6 节的 解释 A vs 解释 B。

### 5.4 本地环境限制

本地 macOS 无 CUDA：
```
CUDA: False
Device count: 0
```

baseline checkpoint (`work_dirs/crane_symeood_k1/epoch_24.pth`) 和测试数据都在本地，但 probe 需要在远程 GPU 机器跑。injector 模型的 checkpoint 也不在本地（`work_dirs/` 只有 `crane_symeood_k1`）。

---

## 六、决策框架

```
                    P2 vs P3 比较
                   /              \
            P3 > P2               P3 ≈ P2 或 P3 < P2
           injection 帮了          injection 没帮或伤了
              |                        |
         P1 vs P2                   Path A 死
        /        \               考虑路径 B（scoring head 调制）
   P1 ≈ P2    P1 ≠ P2           或路径 C（重新审视 re-score 边界）
      |          |
   backbone    backbone
   没变        变了
      |          |
   A1 先行     A1 先行
   (ordinary   (ordinary
   + gate)     + gate)
      |          |
   有效?      有效?
   / \         / \
  是  否      是  否
  |   |       |   |
 A2  考虑   A2  考虑
(strong  路径B  (strong  路径B
+consist)或C   +consist)或C
```

### 推荐执行顺序

1. **修 probe**（加 `--apply-injection` flag）—— 10 行代码
2. **跑 P1**（baseline, 无 injector）—— 需远程 GPU, ~5 分钟
3. **跑 P2 + P3**（ordinary injector, 不 inject / with inject）—— 需远程 GPU + injector checkpoint, ~10 分钟
4. **per-sequence MCML 分析**（ordinary + strong test 结果）—— 纯数据分析, 无需 GPU
5. 用判据表决策
6. 如果 α 或 δ：实现 A1（ordinary + gate），训练评估
7. 如果 A1 有效但不够强：实现 A2（strong + gate + L_ctx-consist）
8. 如果 β 或 γ：放弃 Path A，转路径 B 或 C

---

## 七、路径 A 之外的方向（如果 Path A 被否决）

### 路径 B：scoring head 调制

不调 FPN feature，而是让平台 context map 直接作用在 cls 分支的中间表征上。modulation 直接打在问题层（scoring），而非间接通过 feature。

需确认：是否触碰「不能做 re-score」边界。我的理解是——re-score 是推理期用平台重新评估置信度排序候选，而这里是训练时用平台 context 塑形 scoring head 的特征表征，推理时 scoring head 已经内化了平台信号。两者不同。

### 路径 C：重新审视 re-score 边界

原设计 §12 的 MVP 正是「非分数播种 + 外扩头兼做上下文 re-score」。如果这条被过早封死，可能关掉了最直接的解法。

需区分：
- **候选过滤式假修复**（应禁止）：没有真框、造一个假框塞进去
- **上下文 re-score**（可能合理）：好框一直在、只是分数不对，用平台上下文重新评估置信度

后者不是造假——它是在用额外信息修正一个已知崩溃的打分通道。

---

## 附录：关键文件路径

| 文件 | 作用 |
|---|---|
| `mmrotate/models/detectors/sym_eood_detector.py` | 检测器，simple_test/forward_train |
| `mmrotate/models/dense_heads/platform_context_injector.py` | Stage2 injector |
| `mmrotate/models/dense_heads/platform_context_head.py` | Stage1 辅助头 |
| `crane_project/configs/crane_symeood_k1_platform_injector.py` | ordinary 配置 |
| `crane_project/configs/crane_symeood_k1_platform_injector_strong.py` | strong 配置 |
| `crane_project/tools/ctx_entry_probe.py` | hard-slice 入口诊断（需修） |
| `crane_project/tools/temporal_reanchor_probe.py` | 时序重锚诊断 |
| `crane_project/tools/mcml_diag.py` | MCML 诊断 |
| `work_dirs/crane_symeood_k1/mcml_audit/failure_segments.json` | baseline 死段数据 |
| `work_dirs/crane_symeood_k1/mcml_audit/reanchor_aux1_seq02_129_172_oracle_start/summary.json` | 时序重锚数据 |
