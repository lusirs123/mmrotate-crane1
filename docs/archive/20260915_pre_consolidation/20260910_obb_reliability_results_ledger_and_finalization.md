# OBB 分量可靠性结果台账与整体收尾

更新时间：2026-09-10

## 已冻结的研究边界

小论文链路为：复杂港口图像序列 → Base V3 旋转检测 → OBB 中心、尺度、角度分量可靠性判定 → 带 `measurement`、`prediction`、`unavailable` 状态的连续视觉观测。

检测前端使用 Base V3 epoch9：正式 K1 有框时作为锚框，K1 缺失时使用冻结 DINO 候选。Base V3 refiner 输出的固定分数 `1.0` 不作为置信度。

## 已确认的开发结果

- V5.1 在 source-val 上完成冻结，中心使用锚框置信度拒绝，尺度使用 V5 单尺度风险，角度使用置信度并允许一帧保持。它保留为分量策略参考，不声明为总体最优方法。
- V5.2 正式报告 SHA256 为 `ac85dc4c72499645dcc26d30c55f7e3f00bdf5b4fc9081e426c32723fed1e9b0`，决策为 `STOP_V52_PRIMARY_AND_REPORT_ABLATIONS_ONLY`。
- V5.2.1 正式报告 SHA256 为 `800f909cc72facdaecd4089cdb0205a11908b4935b2c4c605873ce78f46d9ce9`。尺度保持补充 19 帧，其中 2 帧正确、17 帧错误，决策为 `DISABLE_V52_SCALE_HOLD_AS_VALID_OBSERVATION`。
- V5.3 正式报告 SHA256 为 `8f3d210ee1e657b7aadd5bb9691985e1e6240fab8ab2c200770ff0fd89466618`。real 域相对已有尺度风险仅 1/5 个覆盖率点胜出，三折只有一折胜多于负，决策为 `STOP_IMAGE_QUALITY_CANDIDATE`。

## 最终比较与接口

小论文固定报告三个方法：Base V3 原始输出、简单置信度拒绝、V5.1 分量策略。V5.2、V5.2.1 和 V5.3 作为消融与失败边界。

在线接口为 `base_v3_obb_paper_runtime_v1`。每帧保留检测来源、分数、原始 OBB，以及三个方法的中心、尺度、角度数值、有效性、状态、来源、历史年龄、风险和拒绝原因。GT 不进入在线接口。

整体报告为 `base_v3_obb_paper_final_report_v1`，绑定正式 V5.1 TEST 报告、V5.1 诊断和 V5.3 停止结果，统一输出分量误差与覆盖率、拒绝恢复、联合有效性、角度保持、尺度同覆盖率比较和完整 OBB RIoU。生成报告前必须由统一在线运行时逐帧复现正式 V5.1 输出，且完整 OBB 重组保留 Base V3 原始宽高与角度轴对应关系。不含检测器和图像 I/O 的观测层运行开销单独打印到终端，不写入需要保持稳定哈希的正式 JSON，也不能表述为端到端 FPS。

fixed TEST 已经暴露，因此整体报告可作为固定划分的可复现实验结果，不作为全新独立序列证据，也不再用于选择阈值、特征、策略或 checkpoint。

## 后续大论文接口

大论文可以消费各分量的 `value`、`state`、`source` 和 `age_since_measurement_frames`，并使用时间戳与图像尺寸字段连接 Raw-opt 深度和状态估计。当前风险值尚未校准为概率或测量方差，不能直接作为 EKF 协方差。

## 从图像到连续观测的统一入口

新增 `base_v3_obb_true_online_pipeline_v1`，把此前分开的两段正式串联为一个按序推理入口：RGB 图像先经过冻结 native-S14 DINO，再把当前 DINO Top-1 与最近四帧严格因果 DINO 历史交给 Base V3 epoch9；Base V3 内部从共享 FPN 特征运行正式 SymEOOD K1 epoch24，以 K1 有框优先、否则 DINO 回退的规则形成锚框并输出 Top-1 OBB；最后把 K1、DINO、锚分数和 Base V3 OBB 送入冻结 V5.1，输出中心、尺度、角度的 `measurement`、`prediction` 或 `unavailable`。

该入口的 `execution_mode` 为 `true_online_model_inference`，不读取缓存 DINO 预测、缓存 Base V3 预测、标注或 GT。序列和帧号只用于排序与间断复位，不作为检测或可靠性路由。Base V3 新增的 `last_inference_records()` 只导出已在同一次 forward 中计算的 K1/DINO/锚来源证据，不重复运行 K1，也不改变模型输出。

可选的 `--paper-final-report` 只在全部在线输出完成后逐帧核对冻结报告中的 Base V3 几何和三分量状态；它不参与任何在线决策，也不授权使用已暴露 TEST 调参。现有 `fixed_test_obb_paper_final_report_v1` 仍是论文固定指标来源；新入口负责证明这些检测与观测模块能够按设计真实串联。

## 真实在线输出的 V2 收尾规则

完整 992 帧在线推理生成后，使用 `base_v3_obb_true_online_finalization_v2` 在推理结束后附加标注。该入口直接计算真实在线输出的中心、尺度、角度、联合有效性和完整 OBB RIoU，并把结果与历史缓存输入生成的 V1 报告逐项比较。它还分别审计 DINO、K1 和 Base V3 几何，因此候选输入变化与四帧历史传播不会再被合并成一个最大数值差。

V2 同时导出不含 GT 的逐帧观测 CSV，保留方法、分量、数值、状态、来源、帧龄、风险、门控和原因，可作为后续 Raw-opt 与状态估计接口。固定 TEST 已经暴露，V2 只用于最终复现和差异归因，不允许据此修改阈值、特征、策略、epoch 或 checkpoint。历史 V1 报告继续保留；若 V2 与 V1 存在输入差异，论文正文应以真实在线结果作为系统实现结果，并把 V1 标为缓存回放参照。

## 真实在线论文指标 V3

新增 `base_v3_obb_true_online_paper_metrics_v3`，只读取已经冻结的 V2 真实在线评估与 V1 在线推理报告，不重新执行检测、拟合阈值或选择策略。最终 JSON 固定输出三张可制表数据：完整 OBB 指标（3 方法 × all/real/sim，共 9 行）、分量指标（3 方法 × 3 分量 × all/real/sim，共 27 行）和锚来源指标（3 方法 × K1/DINO fallback，共 6 行）。补充诊断包括尺度同覆盖率排序、角度保持收益与代价、拒绝恢复以及 V5.1 相对简单置信度拒绝的差值。

V3 同时绑定真实在线报告、V2 后验指标和无 GT 观测 CSV 的 SHA256，并显式检查 992 帧集合、域/序列计数、冻结评估阈值、前端顺序和 Base V3 输出分数语义。`base_v3_output_score=1.0` 被记录为 `constant_refiner_output_not_confidence`，不能进入置信度比较。时间戳在现有输入中为空，报告为 `UNAVAILABLE_NOT_SYNTHESIZED`，不伪造时间信息。

该报告的“完整”仅指固定 TEST 上从 RGB、融合检测到连续 OBB 观测的指标链和论文表格字段已经齐全。它不表示 V5.1 是总体赢家，也不补足未知新序列、物理状态或部署 FPS 证据。fixed TEST 已暴露，V3 之后禁止继续根据该报告改阈值或策略。

