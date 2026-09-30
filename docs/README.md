# 研究文档入口

先看下表的主文档，再按需要查阅专项实验记录。带日期的历史记录保留原始实验细节；同一问题有后续裁决时，以较新记录及其证据身份为准。文档整理没有重新训练、推理或核验外部产物。

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| 冻结 DINOv2 与 SymEOOD 检测 | [检测融合主文档](dino/冻结DINOv2与SymEOOD检测融合.md) | 方法、数据协议、结果、模型身份、复现入口；文末并入语义救援技术记录 |
| OBB 观测可靠性 | [连续输出主文档](obb/OBB观测可靠性与连续输出.md) | Base V3、V5.1–V5.3、在线接口、指标与边界 |
| Webots 单目深度 | [深度主文档](webots/Webots单目深度估计.md) | 几何契约、标定、数据、结果与后续条件 |
| Base V3 + V5.1 小论文 | [完整流程与归档](obb/base_v3_v51_focused_paper_complete_pipeline_20260915.md) | 模型链、固定 TEST 结果、复现命令；文末并入 V3.1 统计修正和交接状态 |
| DINO → SymEOOD 蒸馏 | [蒸馏实验记录](dino/DINO到SymEOOD蒸馏实验总记录.md) | V1–V5、K1 同口径比较、warmstart V2、配对审计与迁移快照 |
| 跨主线测量审计 | [Fixed TEST 测量有效性与 DINO 路线](dino/measurement_validity_and_dino_roadmap_20260919.md) | 测量有效性、候选归因、优化门槛与证据边界 |
| 2026-09-15 整合核对 | [融合核对表](audit/融合核对表.md) | 原锚点映射、状态纠错和待核验项目 |

[整合前归档](archive/20260915_pre_consolidation)保留 11 份原始 Markdown 和 `manifest.json`，用于来源追溯，不再作为当前结论维护。manifest 中的章节编号和正文哈希只对应首次机械整合时的快照。

`docs/data/` 中的 Webots 数据与 `docs/evidence/` 中的蒸馏证据保持原位。历史来源文件并入现有文档后，正文中的“来源：原 docs/…”用于辨认旧路径，不再是可打开的文件链接。
