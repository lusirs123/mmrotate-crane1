# 当前检测配置

当前固定路线：**R4 ep16 → Midpoint σ1.5 ep20 → 双分支LVSR ep5**。配置本体及全部继承定义均在本目录；先在本机修改、验证，再同步服务器。

## 当前模型

| 用途 | 配置入口 | 公共定义 |
| --- | --- | --- |
| R4 | [crane_sgar_r4_v1.py](crane_sgar_r4_v1.py) | [_base_/crane_sgar_detector_v1.py](_base_/crane_sgar_detector_v1.py) |
| Midpoint | [crane_midpoint_r4_v1.py](crane_midpoint_r4_v1.py) | [_base_/crane_midpoint_head_v1.py](_base_/crane_midpoint_head_v1.py) |
| LVSR | [crane_lvsr_r4_midpoint_v1.py](crane_lvsr_r4_midpoint_v1.py) | [_base_/crane_lvsr_head_v1.py](_base_/crane_lvsr_head_v1.py) |
| 完整流程 | [crane_sgar_midpoint_lvsr_pipeline_v1.json](crane_sgar_midpoint_lvsr_pipeline_v1.json) | [运行说明](README_frozen_pipeline_v1.md) |

## 对照与标定

- EOOD对照：[基础配置](crane_eood_k1.py)、[当前数据划分](crane_eood_k1_port_day2night_v1.py)、[相同尺度增强](crane_eood_k1_port_day2night_aug_b_v1.py)。三份组成必要继承链。
- 大论文Raw-opt标定：[原标定参数](depth_scale_calibration_obb_raw_opt_short_q2_train03_v1.json)，参数保持。

共保留11份Python/JSON配置，不含说明文档。work_dirs只保存权重、运行快照和结果，不供给当前正式配置。

## 删除与历史归档

无用的旧DINO检测配置、source-val诊断配置及可靠性组合配置3份已实际删除；恢复位置见[删除记录](../../docs/detection/配置清理记录_20261101.json)。

旧SymEOOD、八组消融和候选配置16份已移出本目录，保存在[历史归档](../../archives/detection_configs_20261101/README.md)，原始内容和SHA保持，另附2份必要继承依赖。主目录不保留快捷方式。

当前冻结推理入口不依赖归档。确需运行旧训练器或历史来源校验时，先按归档说明恢复相应配置；不会自动回退到归档。清理前的目录说明保存在归档的README_before_archive.md。

本轮删除与归档作用于本机；服务器旧实验源、权重和结果未删除或移动。
