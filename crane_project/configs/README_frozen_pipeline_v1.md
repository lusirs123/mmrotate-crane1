# 当前R4／Midpoint／LVSR正式配置

> **命名统一（2026-11-05）：** 当前完整检测前端称为 **SymEOOD**，其中共享几何分配与回归设计称为 **SGAR**；**R4**为采用四层回归特征塔的结构配置编号，实际检测权重为ep16。论文主线使用“SymEOOD → Midpoint → LVSR”，实现与历史记录继续保留R4及原始路径。旧110等SymEOOD版本须带配置限定；SGAR不是完整前端的同义名称。统一规则见[总览命名约定](../../docs/detection/00_论文总览与阅读导航.md#symeood-sgar-r4-naming)。

2026-11-01整理。本机先修改与验证，再同步服务器；`work_dirs`只存权重、运行快照、日志和结果。原实验配置、来源合同和结果不覆盖。

| 文件 | 内容 |
| --- | --- |
| [crane_sgar_r4_v1.py](crane_sgar_r4_v1.py) | 从原R4展开配置复制整理；模型、主辅助监督、分配、数据管线、优化器和日程保持 |
| [crane_midpoint_r4_v1.py](crane_midpoint_r4_v1.py) | 当前σ1.5四中点头、局部采样、解码边界、训练设置与M20身份 |
| [crane_lvsr_r4_midpoint_v1.py](crane_lvsr_r4_midpoint_v1.py) | 当前双分支LVSR、尺寸交付边界、训练设置与L5描述代表身份 |
| [crane_sgar_midpoint_lvsr_pipeline_v1.json](crane_sgar_midpoint_lvsr_pipeline_v1.json) | 三阶段组合、权重相对路径及SHA、运行环境、显存限制、8帧VAL等价检查样本 |

三份Python配置均采用MMCV的`_base_`继承，公共定义保存在本目录的`_base_`中：

```text
_base_/crane_sgar_detector_v1.py  → crane_sgar_r4_v1.py
_base_/crane_midpoint_head_v1.py  → crane_midpoint_r4_v1.py
_base_/crane_lvsr_head_v1.py      → crane_lvsr_r4_midpoint_v1.py
```

R4入口只覆盖回归塔头类型和运行目录；Midpoint入口覆盖σ1.5及M20身份；LVSR入口声明双分支类型、容量及L5身份。完整值通过继承展开读取，不要求入口重复粘贴全部公共参数。SGAR公共模板是110/SymEOOD设计背景，不是原版EOOD。

模型固定R4 ep16 → M20 → 双分支L5。LVSR原`selected_epoch=null`，L5是原VAL描述代表；配置整理不改变选权或采用沿革。配置中的算法常量与原实现加载时核对，不通过修改全局SETTINGS改变旧模型行为。改变实际结构或算法应作为新的配置版本及相应源码修改，不能只改表述。

## 统一入口

从项目根目录运行。默认检查只读配置和文件SHA，不导入Torch、读取图像或访问GPU：

```bash
python crane_project/tools/run_crane_frozen_pipeline_v1.py --stage check
```

单张图片完整推理：

```bash
python crane_project/tools/run_crane_frozen_pipeline_v1.py \
  --stage predict --gpu 0 --image /absolute/path/image.jpg \
  --out-dir work_dirs/my_frozen_prediction_001
```

与原封存结果进行8帧VAL等价检查（不是重新评价TEST）：

```bash
python crane_project/tools/run_crane_frozen_pipeline_v1.py \
  --stage verify-val --gpu 0 \
  --out-dir work_dirs/my_frozen_val_check_001
```

输出目录必须不存在，不覆盖旧结果。推理只接受图像及其变换信息，不读取GT；VAL参考框仅用于推理后的等价比较。三个权重独立加载，参数/缓冲前后核对；P3由本次R4前向产生，不使用ROI缓存冒充完整推理。不调用或动态替换旧训练脚本。

本版本绑定历史验证环境Torch1.13.1/CUDA11.7/cuDNN8500、GTX1080、FP32/batch1；其他环境需另做兼容验证，不自动放宽一致性要求。运行前需有4GiB空闲，分配限额3GiB，进程驱动显存达4GiB或空闲低于1GiB停止。命令示例不授权占用已有任务资源，也不是训练命令。

## 与旧配置的区别

R4旧快照保留在 `work_dirs/symeood_110_reg_tower_20261030_v1/config.py`，仅供历史核对。正式配置的全部内容及继承文件都在`crane_project/configs`，不读取旧快照，也不使用指向旧配置的快捷方式或`artifact_reference`。新配置仅调整运行元信息：清除绝对`load_from`、旧RNG/输入追溯Hook及其导入，使用新的相对`work_dir`。模型、数据及训练配方逐项相同；新配置不是重放旧完整初始状态/RNG的合同。精确历史训练仍查原初始化与封存快照，不能把重新随机初始化训练称为原实验的逐位复现。

Midpoint和LVSR此前没有独立MMRotate检测器配置；本次将散落的参数整理成独立配置，用于统一入口及复现说明。它们仍是分阶段训练的小头，不能直接把头权重交给普通`tools/test.py`。本次没有重写或重跑旧训练器。

旧脚本和来源合同保留历史用途，包括服务器原R4启动脚本与本机回读命令修正的差异；新入口不依赖该脚本，也不要求恢复已清理的无关实验文件。后续训练入口若需统一，须在本机另行适配和验证，不在服务器临时改模型源码。
