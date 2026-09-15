# 深度阶段图形数据清单

当前任务不生成论文图。后续可从 `docs/data/webots_depth_formal_frame_index_v1.csv` 生成深度轨迹、摆角覆盖分布和误差分层图。检测 OBB 误差传播图必须等待同一 fixed-dev 序列上的冻结检测结果，不能使用 GT OBB 指标替代。


## OBB 分量可靠性数据

- `work_dirs/base_v3_obb_reliability_baseline_v1/source_val_component_policy_audit_v4.json`：待服务器生成的 source-val 开发审计，支持 gate 主比较、简单基线、来源分层和连续状态消融。
- `work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_obb_observation_eval_v1.json`：已经冻结的 fixed TEST 诊断，只作为现有 V3 方法结果，不用于 V4 阈值拟合。
- 后续论文图可从 V4 JSON 生成分量 matched-coverage 曲线和 coverage-error 图；图注必须标明 source-val development 或 fixed TEST，不能标为未知序列。
