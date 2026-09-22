# Fixed TEST 测量有效性对照与 DINO 优化路线

更新时间：2026-09-22

本文件保存 2026-09-19 至 2026-09-22 完成的测量有效性审计、DINO 候选追踪、正式 NMS 配对诊断、native spatial adapter 训练结果和文献核验。它是实验裁决记录，不是论文正文，也不授权根据 fixed TEST 继续选阈值或 checkpoint。

## 1. 证据身份与边界

| 证据 | 文件 | SHA256 | 角色 |
| --- | --- | --- | --- |
| 测量有效性审计 | `measurement_validity_audit_v2.json` | `f4a649be00b25d0d66959e289a4b6f8057c9e9d94f02935f34655b6399c28078` | fixed TEST 敏感性分析 |
| native-S14 候选追踪 | `frozen_dino_native_s14_candidate_trace_all_v3.json` | `b3ef8eac9878892b75b29fa72066733c4766ddea8319d4545c5c05c7151d7766` | 三段已暴露困难切片的失败归因 |
| 正式 NMS 配对诊断 | `frozen_dino_native_s14_formal_nms_paired_all_v4.json` | `82b863f60052607697540b080018f0dac9037f2fa9931a947e3f9e2fba08ccb5` | 历史 NMS 0.1 与 source-selected NMS 0.5 的只读对照 |
| spatial adapter 训练 | `native_spatial_adapter_training_v3.json` | `bd964b39f0ba8dadd2c4a1f6a2aef97b11127f5975b7f702372ef65557cd03f4` | source-only 失败实验 |

测量审计的绑定输入为：

- finalization protocol：`base_v3_obb_true_online_finalization_v2`
- finalization SHA256：`1d848597ac1488a4bb70d290b69bdb8ea1405cb8e7f1134f505fe4b9fa0ae88b`
- online pipeline SHA256：`ef5487b9b20cc8f7723fd06f5004b330060f101637b3f68ebdbec23ac1a158ac`
- 抓料/未完全离料区间：`real_seq03[130,187]`，闭区间，共 58 帧
- fixed TEST 总帧数：992；measurement-valid 934 帧

审计文件明确记录 `fixed_test_read=true`、`prediction_or_threshold_selection_performed=false`、`intervals_are_prediction_independent=true`。该测量有效性审计没有训练模型、修改预测、选择阈值或 checkpoint。

## 2. 测量有效性审计结果

### 2.1 完整 TEST 与 measurement-valid 对照

| 方法 | 统计范围 | 帧数 | 中心观测可用 | 中心观测可用率 | RIoU 样本数 | 平均 RIoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | 全部帧 | 992 | 992 | 100.00% | 992 | 0.796358 |
| raw | measurement-valid | 934 | 934 | 100.00% | 934 | 0.816101 |
| raw | 排除区间 | 58 | 58 | 100.00% | 58 | 0.478421 |
| score rejection only | 全部帧 | 992 | 826 | 83.27% | 826 | 0.827773 |
| score rejection only | measurement-valid | 934 | 806 | 86.30% | 806 | 0.834691 |
| score rejection only | 排除区间 | 58 | 20 | 34.48% | 20 | 0.548974 |
| V5.1 hybrid | 全部帧 | 992 | 826 | 83.27% | 787 | 0.832538 |
| V5.1 hybrid | measurement-valid | 934 | 806 | 86.30% | 767 | 0.839932 |
| V5.1 hybrid | 排除区间 | 58 | 20 | 34.48% | 20 | 0.548974 |

排除 58 帧后的平均 RIoU 变化为：raw `+0.019743`，score rejection only `+0.006918`，V5.1 hybrid `+0.007394`。这证明该区间是明显困难区间，但没有证明低质量完全由抓料动作造成。逐帧动作状态和因果关系仍未验证。

正式报告规则：完整 fixed TEST 是主结果；measurement-valid 是预定义操作区间的敏感性分析；两者并列报告，不用排除后的高指标替代完整流程性能。

### 2.2 分量可靠性

| 方法 / 范围 | 分量 | 输出覆盖率 | 正确输出覆盖率 | 错误可用率 | 最长不可用连续帧 |
| --- | --- | ---: | ---: | ---: | ---: |
| raw / 完整 TEST | 中心 | 100.00% | 78.02% | 21.98% | 0 |
| raw / 完整 TEST | 尺度 | 100.00% | 64.82% | 35.18% | 0 |
| raw / 完整 TEST | 方向 | 100.00% | 81.25% | 18.75% | 0 |
| V5.1 / 完整 TEST | 中心 | 83.27% | 70.87% | 14.89% | 19 |
| V5.1 / 完整 TEST | 尺度 | 95.36% | 62.70% | 34.25% | 17 |
| V5.1 / 完整 TEST | 方向 | 91.23% | 75.91% | 16.80% | 18 |
| V5.1 / measurement-valid | 中心 | 86.30% | 73.98% | 14.27% | 11 |
| V5.1 / measurement-valid | 尺度 | 95.07% | 66.38% | 30.18% | 17 |
| V5.1 / measurement-valid | 方向 | 94.00% | 79.76% | 15.15% | 10 |

`center_valid_count` 只表示中心观测可输出，不表示中心正确。V5.1 完整 TEST 的中心可用数为 826，而完整 OBB 的 RIoU 样本数为 787，说明部分帧只有中心或部分分量可用。输出覆盖率、正确输出覆盖率、错误可用率、完整 OBB 可用性和连续缺失不能互相替代。

V5.1 的正确结论仍是：它通过保留部分尺度/方向分量提高了部分观测可用性，但尺度错误可用率仍为 34.25%，并且中心覆盖没有高于 score rejection only。V5.1 是分量级折中方案，不是整体最优方法。

DFR、ACI、MCML 和完整运行时间继续从绑定的 finalization/online 报告提取。本审计只重算上述几何与分量统计，不能替代完整时序报告。

## 3. DINO 当前困难的正式归因

### 3.1 候选存在，但最终选择失败

候选追踪 V3 对三段已暴露困难切片共 137 帧完成了逐帧复现：

- 指标最大绝对复现误差：`2.9206e-06`，低于 `1e-3` 容差；
- 候选计数检查：`548/548` 通过；逐帧重建：`137/137` 通过；
- decode 后存在可用候选：136 帧；历史 NMS 后仍存在可用候选：117 帧；
- 历史失败归因：NMS suppression 19 帧、ROI regression 1 帧、Top-1 success 117 帧；
- 被 NMS 抑制的可用候选累计 2161 个。

这说明主要困难不是 DINO 完全看不到目标，而是可用旋转框经过候选竞争后没有安全成为最终输出。

### 3.2 放宽到正式 NMS 0.5 仍不改善 Top-1

正式 NMS 配对诊断比较历史 `IoU=0.1` 与已经由 source 选择固定的正式 `IoU=0.5`：

| 指标 | 历史 0.1 | 正式 0.5 |
| --- | ---: | ---: |
| Top-1 命中 | 117/137 | 117/137 |
| NMS suppression 失败 | 19 | 8 |
| final ordering 失败 | 0 | 11 |
| ROI regression 失败 | 1 | 1 |
| NMS 后含可用候选帧 | 117 | 128 |

正式 NMS 恢复了 11 帧的可用候选，却没有增加任何 Top-1 命中；这些帧从“NMS 抑制失败”转成“最终排序失败”。`seq03_small` 中，历史的 13 个 NMS 失败变为正式策略下 8 个 NMS 失败和 5 个最终排序失败，Top-1 仍为 `50/64`。

因此：NMS 确实参与失败，但单独调 NMS 阈值不是解决方案。第一优化重点应是让分类/语义分数与旋转框几何质量对齐，再把 NMS 作为固定策略下的消融或轻量补充。

### 3.3 native spatial adapter 没有形成 source-safe 改进

adapter V3 使用官方 source train 2781 帧、source val 738 帧，只在 source 上训练，DINOv2 保持冻结；head 峰值显存约 `371.41 MB`，显存控制符合本轮要求。

正式基线为 full Top-1 `677/738`、small Top-1 `303/350`、MCML `3/3`。四个训练 epoch 都未通过 source exact-retention：

- epoch 1：full `676/738`，旧正确帧丢失 2、获得 1；
- epoch 2–4：full `675/738`，旧正确帧丢失 4、获得 2；
- 最终 `best_epoch=0`，回退原始正式 checkpoint。

这证明同一 stride-14 特征前的轻量残差修正虽然可以稳定训练，却没有解决候选排序，不能作为已经成立的创新或部署升级。它应作为受控失败实验保留。

### 3.4 当前困难总结

1. 小目标的短边接近一个 DINO patch，细粒度几何证据弱；高分辨率 S7 已证明能补候选覆盖，但没有解决 Top-1。
2. 分类分数与 OBB 定位质量错位。正式 NMS 能保留更多正确候选，却不能把它们排到第一。
3. source 中“基线错而新候选正确”的跨序列监督不足，强排序器容易用少量增益换取旧正确帧退化。
4. 逐帧检测恢复后，宽高和周期角度的抖动会影响 DFR/ACI；时序稳定只能处理已经选中的框，不能替代候选选择。
5. ViT-L/14、2000 proposals 和大量 ROI 带来较高推理成本；轻量化必须在检测身份固定后进行，否则无法区分结构改进与压缩损失。

### 3.5 当前数据困难与证据边界

本节记录数据限制；后续训练优先级和完整收益/退化复盘以[统一DINO文档的2026-09-22整体复盘](冻结DINOv2与SymEOOD检测融合.md)为准。数据不足尚未被证明是所有失败的唯一原因，新增数据也不保证排序问题自动解决。继续优先准备检测性能改进，系统轻量化随后进行。

2026-09-22 的服务器实现核验已经通过：

```text
DINO_QUALITY_RANKING_IMPLEMENTATION_OK
verified: roi_pairwise_margin_loss,
          mine_actual_roi_competitor_pairs,
          select_representative_usable_rois,
          pairwise_v2_source_selection_gate,
          build_source_native_quality_feasibility_audit
```

因此，当前数据困难不能再表述为“尚未进行困难候选挖掘”。已有代码和实验已经完成实际 ROI 竞争候选挖掘、代表性可用候选选择、排序损失、跨序列门控和 source 可行性审计；Pairwise V1/V2、连续/相对质量、high-resolution ranker 与 unified hard-pair 也已经覆盖主要候选质量学习路线。继续更换名称重写同类 head、loss 或 margin，不能补足缺失的证据。

真正的数据限制是“帧多、独立困难事件少”：视频相邻帧高度相关，普通帧数量不能等价为独立监督。现有 real source-train `2,033` 帧全部 native 正确，能够形成 natural native-wrong/S7-correct 增益监督的样本只有 `sim_seq08` 的 `7` 帧；同时没有可用于证明跨域接管安全性的独立 real hard validation sequence。旧 validation 中存在自然增益，也不能在不改变既有 source gate 的情况下直接并入训练。

当前数据状态应分成三层理解：

1. **已完成的标注数据内挖掘。** 已有 source 和固定诊断切片上的候选覆盖、NMS 抑制、最终排序、ROI 回归及竞争候选关系已经检查；这些工作不再作为下一阶段重新执行。
2. **尚未确认的剩余视频价值。** 若仍有未扫描视频，只需复用现有入口做一次只读筛查，确认是否包含新的拍摄时段、视角、工况和独立困难事件。来自同一旧视频的连续近重复帧只能增加样本数，不能自动增加跨序列证据。
3. **真正缺少的训练与验证支持。** 后续质量排序若要成立，需要来自多个独立真实序列的困难候选对，并预先按完整序列固定训练与验证角色。该要求不意味着重划旧 train/val/test，也不要求重做历史实验；固定 TEST 继续只承担冻结后的最终评估。

对剩余视频的筛查结果必须据实裁决：若找到新的独立困难事件，才进行稀疏 OBB 标注并生成候选质量对；若只能找到同场景近重复帧，停止继续堆叠同类排序实验，将它们保留为无标注时序材料或运行稳定性数据。不能因为“还有视频”就声称已经解决监督不足，也不能根据已经暴露的 target 切片挑选训练样本、阈值或 checkpoint。

这一限制与当前模型困难是一致的：S7 已把正确小目标候选带入候选池，正式 NMS 也恢复了部分被抑制候选，但现有监督不足以学习一个既提升困难帧、又不破坏旧正确帧的跨域安全排序器。它是当前尚未闭环的数据与验证问题，不是遗漏了一轮候选挖掘实现。

## 4. 优化优先级与实验门槛

### P0：冻结现有正式基线

正式 DINO 基线保持 native S14、ROI classifier `alpha=0.5`、S7 disabled、NMS IoU `0.5`。fixed TEST 和三段困难切片只用于绑定报告与失败归因，不再用于选择阈值、epoch、特征或排序器。

### P1：候选级 OBB 几何质量排序已经做过，不重复实现

仓库已经覆盖了“绝对质量回归”和“候选相对排序”两条路线，不能再以新名称重复实现：

| 已完成实验 | 关键结果 | 裁决 |
| --- | --- | --- |
| native-S14 Pairwise V1 | source 最好 full `679/738`、small `307/350`，但 exact retention 失败；固定 `seq03_small=45/64` | 不采用 |
| native-S14 Pairwise V2 | epoch 1 为 full `674/738`、small `301/350`、旧正确丢失 3；后续更差 | 回退 epoch 0，关闭 |
| 连续 RIoU quality head | full `681/738`、small `306/350`、`lost=0`，未达到绝对门槛 | 回退 epoch 0 |
| relative-quality + student | source teacher 达 `691/738`、`312/350`、`lost=0`；固定 `seq03_small` 仍为 `50/64` | source 有效，target 诊断失败 |
| high-resolution ROI ranker | source `688/738`、`311/350`、`lost=0`；R@100 补到 `64/64`，Top-1 仍 `50/64` | 冻结诊断候选 |
| unified hard-pair ranker | source `696/738`、`320/350`，但 `lost=1` | 仅 bounded-risk 证据 |

因此现有代码已经验证：增加 quality/ranking head 不是当前缺失项。真正缺口是 source-train 中跨 real/sim、跨 sequence 的自然增益监督，以及独立 real hard sequence 的留出验证。现有 real source-train 2033 帧全部 native 正确，natural native-wrong/S7-correct 只有 `sim_seq08` 的 7 帧；在这种支持下继续增加 head、loss 或 margin，不能证明跨域安全接管。

下一步前置条件是补充至少两个独立、带 OBB 标注的真实困难连续序列，并在读取结果前按完整序列固定 train/validation 角色。这里不是重划现有 train/val/test，也不要求重做历史实验；新数据只用于后续排序器的训练支持和独立验证。

### P2：只对仍缺候选的区域使用稀疏高分辨率

正式 NMS 下已有 128/137 帧保留可用候选，说明高分辨率不是所有失败的第一需求。仅在 source 证明候选覆盖仍不足的样本上启用稀疏 stride-7/高分辨率读出，并与候选质量排序分开消融。

### P3：排序通过后再做因果时序稳定

时序模块只允许处理 source-safe Top-1 的中心、尺度、周期角度和 DINO ROI embedding。它不能补造不存在的候选，也不能通过未来帧或 target 标签选择当前输出。先证明单帧排序，再评估 DFR、ACI 和 MCML。

### P4：模型身份固定后轻量化

顺序为：减少 proposal/ROI 数量并做分阶段计时；稀疏高分辨率；小型质量头；再考虑从 ViT-L/14 蒸馏到 DINOv2 ViT-B/14 或 ViT-S/14。每一步同时报告参数量、峰值显存、端到端 FPS、DINO 前向、RPN/ROI 时间和 source-retention，不能只报告 head 显存。

## 5. 可用文献与本项目映射

本轮用四个主题检索主干、自监督冻结特征、小目标高分辨率、候选排序/NMS 和轻量化，共审阅 81 条检索候选；去重后只保留与当前已确认失败机制直接相关的原始论文。文献用于设计依据，不是本项目实验结果。

| 文献 | 原论文支持的内容 | 对本项目的直接用途 | 不能据此声称 |
| --- | --- | --- | --- |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 大规模自监督 ViT 可提供可迁移的冻结图像与 patch 特征，并蒸馏出较小模型 | 继续复用冻结 DINOv2；后续可评估 B/14、S/14 学生 | 不能证明当前抓斗 OBB 头已经最优 |
| [DINO Teacher, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lavoie_Large_Self-Supervised_Models_Bridge_the_Gap_in_Domain_Adaptive_Object_CVPR_2025_paper.html) | 在冻结 DINOv2 上用 source 标注训练 labeller，并在完整方法中加入 target feature alignment/student | 支持本项目 source-only frozen-DINO labeller 的出发点 | 本项目未实现其 target pseudo-label student 和完整 alignment，不能称为完整复现 |
| [Generalized Focal Loss, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html) | 把分类与定位质量合并为联合连续表示，减少训练与推理评分不一致 | 候选质量分数的基础实现参照 | 水平框结果不能直接等同于旋转 OBB 收益 |
| [VarifocalNet, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html) | IoU-aware classification score 改善大量候选的排序 | 直接对应“候选存在但 Top-1 错”的问题 | 不能照搬完整 dense detector；应只实现小型 ROI 质量读出 |
| [Rank-DETR, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html) | 分类置信与定位质量错位会损害 Top-ranked box，排序损失可优先高质量框 | 支持 hard-pair/relative ranking 的对照实验 | DETR query 结构不是当前 RPN/ROI 结构的直接模板 |
| [Soft-NMS, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html) | 用重叠相关的连续衰减替代直接删除候选 | 可作为固定 baseline 后的低成本 NMS 消融 | 本项目配对实验已表明仅保留候选不会自动改善 Top-1 |
| [QueryDet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) | 低分辨率先定位粗区域，只在稀疏位置计算高分辨率特征，可兼顾小目标与速度 | 支持后续稀疏 S7/ROI 计算和轻量化 | 当前候选覆盖已较高，不能先于质量排序全面增加高分辨率计算 |
| [Pixel-level Quality Assessment for Oriented Object Detection, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38411) | 旋转框分类分数与定位质量可能错位；像素级空间一致性可用于 OBB 质量估计 | 最贴近当前 OBB 候选质量问题，可作为质量读出的高成本上界 | 既有 BrightAug PQA 实验失败，不能直接授权再次实现相同 PQA；先做最小 ROI 质量预检 |

综合判断：文献支持“冻结 DINO 表征 + 小型 OBB 候选质量排序 + 必要时稀疏高分辨率 + 后续蒸馏”的路线。创新不能写成“使用 DINOv2”，而应建立在抓斗旋转框的几何质量排序、source exact-retention、安全连续输出和资源约束共同成立之后。

## 6. 下一步可执行的最小闭环

### 6.1 有新独立真实序列时

1. 补充至少两个独立真实困难连续序列，覆盖远距、小目标、抓料前后和正常背景；按完整序列预注册 development-train 与 held-out validation，不随机拆相邻帧。
2. 复用现有候选导出、Pairwise V2、continuous/relative quality 和 exact-retention gate，不新建同构排序器。
3. 先运行只读支持审计：要求 native-wrong/new-candidate-correct 与 native-correct/new-candidate-wrong 均在多个真实序列出现；支持不足则停止训练。
4. 支持充分后只选一个现有最小排序入口重训；选择报告必须包含 full/small、real/sim、逐序列 Top-1/RIoU、old-correct lost、MCML、DFR、ACI、参数量、峰值显存和耗时。
5. source 与新 held-out real gate 同时通过后冻结模型；fixed TEST 只运行一次报告，不反向选参数。

### 6.2 暂时没有新数据时

不再训练质量头、扫描 margin、改变 NMS 或读取 fixed TEST。可继续的工作只有：

1. 保持正式 native-S14 `alpha=0.5`、S7 disabled、NMS `0.5`；
2. 完成现有 seq11 251 帧的时序块 OOF/CV，且只解释为同视频机制证据，不称为未知序列泛化；
3. 开展不改变检测身份的运行时间分解与 proposal/ROI 数量离线敏感性审计，为后续轻量化确定预算；
4. 等检测方法由独立数据固定后，再开展 ViT-B/14 或 ViT-S/14 蒸馏。

当前最重要的缺口是数据证据，不是新的网络模块。继续在相同 source 与已暴露 TEST 上叠加排序头，只会重复已经完成的实验族。

## 7. 当前结论

- 抓料区间审计已经完成，其作用是划清测量有效性边界，不是提高模型性能。
- DINO 的正式困难已经收窄为：小目标细粒度不足、分类与定位质量错位、source 安全监督不足，以及后续推理成本偏高。
- 正式 NMS 0.5 恢复了更多可用候选，却没有提高 Top-1，证实下一步应优化候选质量排序。
- native spatial adapter V3 没有通过 source-retention，不能写成已成立创新。
- 当前正式部署基线不变；候选质量排序实验族已经完成。下一项模型训练必须等待新的独立真实困难序列提供跨序列监督与验证支持；无新数据时只做 seq11 OOF 机制核验和轻量化预算审计。
