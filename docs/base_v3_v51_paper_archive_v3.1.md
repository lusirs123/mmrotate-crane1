# Base V3 + V5.1 小论文归档 V3.1

**归档日期：** 2026-09-15  
**正式 JSON：** `crane_project/data_contracts/base_v3_obb_paper_archive_v31.json`

## 修正内容

尺度同覆盖率比较此前使用 `<=` 判断双指标胜出，因此当 learned scale risk 与 anchor score 的均值误差、失败率完全相等时，也会被计入 `risk_wins_both_count`。V3.1 将统计改为：

- 两项指标都严格改善（`<`）才计为双指标胜出；
- 均值误差相等和失败率相等分别记录；
- 两项同时相等时记录为 `tie_mean_and_failure`，并汇总 `matched_coverage_tie_count` 与 `matched_coverage_tie_rate`；
- 报告写明 `tie_policy: strict_metric_improvement_required`。

该修正只影响统计解释，不改变冻结 TEST 样本、阈值、排序输入或 V5.1 在线策略。

## 正式产物

- 实现：`crane_project/tools/base_v3_obb_hybrid_fixed_test_diagnostic_v51.py`
- 诊断契约：`crane_project/data_contracts/base_v3_obb_hybrid_fixed_test_diagnostic_v51.json`
- 论文汇总契约：`crane_project/data_contracts/base_v3_obb_paper_final_report_v1.json`
- 回归测试：`tests/test_base_v3_obb_hybrid_fixed_test_diagnostic_v51.py`

## 验证

```text
python -m pytest -q tests/test_base_v3_obb_hybrid_fixed_test_diagnostic_v51.py tests/test_base_v3_obb_paper_finalization_v1.py
10 passed
```

## 证据边界

V3.1 是已暴露固定 TEST 上的归档统计修正，不产生新的泛化、校准、物理状态或部署性能结论。
