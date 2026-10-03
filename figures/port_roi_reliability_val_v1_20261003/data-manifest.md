# VAL图表数据清单

真实服务器回传数据；固定B ep24、质量分支ep08，单seed探索性VAL。

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| Component risk–coverage | component_coverage_all_methods.csv | Real | 原val_compare.json（SHA见JSON） | summarize_val.py | fig1_component_risk_coverage.png / .svg |
| Correct component availability | component_coverage_all_methods.csv | Real | 同上 | summarize_val.py | fig2_correct_component_coverage.png / .svg |

全表含四方法与六个分组；主图仅score/geometry/ROI的两域结果。无mock数据，无新模型运行，无TEST评价或阈值选择。
