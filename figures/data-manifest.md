# 深度阶段图形数据清单

当前任务不生成论文图。后续可从 `docs/data/webots_depth_formal_frame_index_v1.csv` 生成深度轨迹、摆角覆盖分布和误差分层图。检测 OBB 误差传播图必须等待同一 fixed-dev 序列上的冻结检测结果，不能使用 GT OBB 指标替代。


## OBB 分量可靠性数据

- `work_dirs/base_v3_obb_reliability_baseline_v1/source_val_component_policy_audit_v4.json`：待服务器生成的 source-val 开发审计，支持 gate 主比较、简单基线、来源分层和连续状态消融。
- `work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_obb_observation_eval_v1.json`：已经冻结的 fixed TEST 诊断，只作为现有 V3 方法结果，不用于 V4 阈值拟合。
- 后续论文图可从 V4 JSON 生成分量 matched-coverage 曲线和 coverage-error 图；图注必须标明 source-val development 或 fixed TEST，不能标为未知序列。

## 2026-10-03：当前B＋ordinary ROI的已有VAL图表

本节为当前已生成图表；上方深度任务及历史V3/V4记录保留其原身份。只复用固定B ep24与分支ep08的已有887帧缓存VAL评分，未新增拟合、模型推理或TEST读取。

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| 分量错误率—接受覆盖 | `port_roi_reliability_val_v1_20261003/component_coverage_all_methods.csv` | Real | `work_dirs/port_reliability_branches_v1_server_review_20261003/val_compare.json`，输入SHA见图表子目录data-manifest.json | `port_roi_reliability_val_v1_20261003/summarize_val.py` | `fig1_component_risk_coverage.png` / `.svg` |
| 正确接受分量的全资格帧覆盖 | 同上 | Real | 同上 | 同上 | `fig2_correct_component_coverage.png` / `.svg` |

图表位于该子目录，450dpi PNG及SVG均已生成并目视检查。原始覆盖CSV含四方法、六分组、六预定档和三分量共432记录；主图展示score/geometry/ROI的两域结果。曲线是单seed、已暴露VAL上的同接受数量离线排序评价，不是部署阈值、未知视频泛化或检测框几何提升。
