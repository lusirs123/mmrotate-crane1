# 深度实验表格数据结构

| 表格 | 用途 | 行 | 指标 | 数据来源 |
| --- | --- | --- | --- | --- |
| Webots 数据集构成 | 说明协议划分和覆盖 | Train-03、Fixed-dev-01、Unknown-02 | 帧数、FPS、深度范围、摆角范围 | sequence manifest + frames.jsonl |
| 深度公式比较 | 选择冻结公式 | M0、M1、M1S、M2S、M2 | MAE、RMSE、AbsRel、Bias、P95、Max | Fixed-dev-01 真值 OBB |
| 未知序列泛化 | 评估冻结公式 | 全部、摆角分层 | MAE、RMSE、AbsRel、P95 | Unknown-02 真值 OBB |
| 检测误差传播 | 分离检测与公式误差 | GT OBB、各冻结检测模型 | OBB 几何误差、深度误差、摆角误差 | 待生成同帧检测 JSONL |

| OBB gate 主比较 | 比较统一置信度、分量风险和简单可用性规则 | raw、score、component、feature availability、K1-only | 分量平均/尾部误差、错误率、matched coverage | source-val V4 policy audit JSON |
| OBB 连续状态消融 | 分离拒绝与短时保持的贡献 | 各 gate 的 rejection-only、bounded-hold | measurement/output/correct coverage、prediction delta、最长 unavailable | source-val V4 policy audit JSON |
| OBB 来源与序列分层 | 定位 K1 与 fallback 的适用边界 | real、sim、逐序列、K1、DINO fallback | 各分量误差、覆盖率、支持数 | source-val V4 policy audit JSON |
| OBB 最终泛化 | 验证冻结策略的跨视频行为 | 最终冻结策略、强简单基线 | 相同指标和视频块置信区间 | 待采集或冻结的独立视频 |
