# Fixed TEST 测量有效性对照与 DINO 后续路线

## 1. 证据身份

- measurement-validity protocol：`base_v3_obb_measurement_validity_audit_v1`
- 输入报告 protocol：`base_v3_obb_true_online_finalization_v2`
- 抓料区间：`real_seq03[130,187]`，共 58 帧
- fixed TEST 总帧数：992
- measurement-valid 帧数：934
- measurement-invalid 帧数：58
- 审计不修改原始报告，不参与训练、阈值选择或 checkpoint 选择。

审计文件：`measurement_validity_audit_v1.json`。文件中的输入 SHA256 是正式归档身份的一部分。

## 2. raw 与 measurement-valid 对照

| 方法 | 统计范围 | 帧数 | 中心观测可用 | 中心观测可用率 | RIoU 样本数 | 平均 RIoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | 全部帧 | 992 | 992 | 100.00% | 992 | 0.79636 |
| raw | measurement-valid | 934 | 934 | 100.00% | 934 | 0.81610 |
| raw | 抓料区间 | 58 | 58 | 100.00% | 58 | 0.47842 |
| score rejection only | 全部帧 | 992 | 826 | 83.27% | 826 | 0.82777 |
| score rejection only | measurement-valid | 934 | 806 | 86.30% | 806 | 0.83469 |
| score rejection only | 抓料区间 | 58 | 20 | 34.48% | 20 | 0.54897 |
| V5.1 hybrid | 全部帧 | 992 | 826 | 83.27% | 787 | 0.83254 |
| V5.1 hybrid | measurement-valid | 934 | 806 | 86.30% | 767 | 0.83993 |
| V5.1 hybrid | 抓料区间 | 58 | 20 | 34.48% | 20 | 0.54897 |

measurement-valid 相对于 raw 的平均 RIoU 变化为：

- raw：`0.79636 → 0.81610`，增加 `0.01974`；
- score rejection only：`0.82777 → 0.83469`，增加 `0.00692`；
- V5.1 hybrid：`0.83254 → 0.83993`，增加 `0.00739`。

这些变化说明抓料或未完全离料阶段是几何质量较低的操作阶段。它们属于测量有效性分层结果，不能解释为模型优化带来的提升。raw 全帧结果仍然是完整作业流程的系统性能依据。

`center_valid_count` 只表示中心误差可以计算，不表示中心已经正确。V5.1 的中心观测数为 826，但 RIoU 可计算数为 787，说明部分帧只有中心分量可用，尺度或方向不足以构成完整 OBB。论文中必须分别报告中心可用、完整 OBB 可用和 RIoU 样本数。

DFR、ACI、MCML 应从 `base_v3_obb_true_online_finalization_v2` 原始报告提取；本审计工具只重算几何核心指标，不能替代完整时序报告。

## 3. DINO 改进能否作为小论文创新

当前不能把 native spatial adapter 写成已经成立的主要创新。它已经证明：

- DINOv2 backbone 可以保持冻结；
- stride-14 特征前加入轻量残差 adapter 可以正常反向传播；
- 显存峰值保持约 371 MB 的 head 开销；
- 但 source validation 上没有提升，并且丢失了旧正确帧，因此严格 source-retention gate 拒绝了所有训练 epoch。

它适合作为受控失败实验，说明“同 stride 特征修正”不能自动解决小目标候选排序问题。

更有论文价值的方向是：在冻结 DINOv2 和已有 OBB 检测器的条件下，针对已经存在于候选池中的正确旋转框，学习一个候选级几何质量排序，并要求旧正确帧零丢失。这个问题与已有的 proposal ranking、localization-aware calibration 和 oriented-box quality assessment 方向相关，创新必须来自本项目的抓斗 OBB、可靠观测分量和连续输出约束，而不是单独声称使用 DINO。

仓库此前已经实现和测试过 S7 high-resolution、relative-quality、affine merge、lane arbitration、temporal association 和 quality suppression。它们说明候选覆盖提升与最终 Top-1 安全接管是两个问题。后续不应继续堆叠同构的全局 boost、margin 或排序器；应从已经通过 source exact-retention 的候选级结果中收束出一个最小方案，并在独立 source 证据下重新确认。

## 4. 后续优化顺序

1. 保留当前 native baseline 作为正式部署候选，保留 adapter v3 作为失败但可复现实验。
2. 对已通过 source-retention 的 high-resolution/relative-quality 候选做统一 provenance 和逐帧复核，确认收益是否来自困难小目标，而不是原本已经正确的帧。
3. 只保留一个候选级质量排序入口，使用 full/small Top-1、RIoU、MCML、DFR、ACI、旧正确帧丢失和多序列净增益共同选模。
4. source gate 通过后固定 checkpoint，再对 fixed TEST 做一次 raw 与 measurement-valid 并列报告。
5. 轻量化在模型身份固定后进行，不能在 TEST 上选择轻量化配置。

## 5. 轻量化路线

优先级从低风险到高风险如下：

1. 保持 DINOv2 ViT-L/14 不变，减少候选数量、ROI 数量和 high-resolution 分支计算，先测量速度与 Top-1/RIoU 变化。
2. 对冻结 DINO 特征做离线缓存或批量提取，区分端到端延迟和检测头延迟；论文中分别报告。
3. 用冻结 ViT-L/14 作为 teacher，蒸馏到 DINOv2 ViT-B/14 或 ViT-S/14，先在 source train/val 进行选择，保留相同候选和可靠性指标。
4. 将候选质量排序器限制为小型 MLP 或低维 ROI readout，禁止再次引入大规模 backbone 或多路重复 ROI 特征。
5. 只有轻量模型通过 source-retention、small-object 和多序列 gate 后，才进行 fixed TEST 测量。

轻量化实验必须同时报告：参数量、峰值显存、端到端 FPS、DINO 特征提取时间、RPN/ROI 时间、raw 与 measurement-valid 指标，以及与 ViT-L/14 baseline 的旧正确帧保留情况。

## 6. 当前结论

抓料阶段分析已经形成有效的敏感性证据，但没有替代完整 raw 结果。DINO adapter 已完成可审计实现，却没有取得 source-safe 性能增益。下一阶段应优先收束已有候选级安全排序路线，再进行轻量化，而不是继续扩大 adapter 或在 fixed TEST 上调参。
