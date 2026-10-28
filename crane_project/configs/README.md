# crane_project/configs：保留范围与Git恢复

整理日期：2026-10-28。本目录由33份配置精简为23份；只删除确认退出当前流程的10份配置及可再生成的字节码／系统杂项。保留配置的路径、内容与SHA均未改变，没有重新编写或重命名训练配置。

## 当前优先使用

| 用途 | 配置／身份 |
| --- | --- |
| 当前110底座的精确运行配置 | `work_dirs/symeood_110_retrain_seed0_20261026_v1/configs/Coff_seed0.py`，位于项目根目录下，沿用原展开配置与环境身份；不因本次清理替换 |
| 110结构配置入口 | `crane_symeood_k1_ablation_110_v1.py`及其继承闭包；不能把默认work_dir当成当前已选权重身份 |
| 原SymEOOD＋尺度增强B | `crane_symeood_k1_port_day2night_aug_b_v1.py`及其父配置；当前下游原生读取仍有依赖 |
| EOOD与相同尺度增强控制 | `crane_eood_k1.py`、`crane_eood_k1_port_day2night_v1.py`、`crane_eood_k1_port_day2night_aug_b_v1.py` |
| 既有完整八组消融 | `crane_symeood_k1_ablation_000/001/010/011/100/101/110/111_v1.py`和公共base；用于既有消融与现有工程检查，不代表授权重训 |
| 原Raw-opt标定 | `depth_scale_calibration_obb_raw_opt_short_q2_train03_v1.json`；原公式、参数和SHA不改 |

当前小论文使用110 ep20 → midpoint ep21 → 双分支LVSR ep11；大论文只将尺寸头换为共享LVSR ep11。midpoint／LVSR按既有专用训练器和合同运行，不是从本目录挑一个新的检测配置重新训练。详见[检测论文总览](../../docs/detection/00_论文总览与阅读导航.md)、[大论文固定身份](../../docs/detection/05_大论文_检测观测基底冻结.md)。

## 为什么另外几份旧配置仍保留

- `crane_symeood_k1_port_day2night_center_size_d_v1.py`、`crane_symeood_k1_port_day2night_shape_e_h_v1.py`、`crane_symeood_k1_port_day2night_size_f_s_v1.py`及`crane_symeood_k1_port_day2night_size_f_s_v1_formal.py`仍被当前midpoint／LVSR冻结来源清单校验SHA，同时被既有针对性检查读取。它们不是当前采用的模块，但删除会破坏当前入口的来源验收，故保留原字节。
- `crane_symeood_formal_dino_native_s14_v1.py`、`crane_symeood_k1_source_val_eval.py`及`base_v3_obb_focused_paper_pipeline_v1.json`仍被独立的旧DINO／可靠性入口或回归检查直接读取。保留这些入口不表示将DINO并回当前小论文／大论文新检测链路。
- `crane_symeood_k1.py`等父配置不能因名字旧而删除；多个现用配置仍继承它们。完整八组仅有8个很短的开关入口，现有消融检查需要它们全部存在。

## 已删除配置及恢复依据

所有删除前的配置都与已提交版本逐字节一致，恢复提交为：

`030395be620d743b64db3bb94bb9ae8d8eb05828`

| 删除的相对文件名 | 原因与旧入口边界 | 原始SHA256 |
| --- | --- | --- |
| `crane_baseline.py` | 旧数据划分的RetinaNet草案；不是当前同划分外部对照结果，当前代码无运行引用。 | `be585172f43a689941ff38a612518be4cdae5762392ce925ea3f7cad8d42c852` |
| `crane_symeood_m2.py` | 早期m2实验，仅在旧ckpt_sweep命令示例中出现。 | `639be2244086716588d2305647abab19f569490aad01afbaa22a359124e1c291` |
| `crane_symeood_k1_brightaug.py` | 旧亮度候选，仅在旧ckpt_sweep命令示例中出现。 | `9006ce61c5a00e68e3ddcf71fb3f4235e3da6f9d5b07269f188224c82d916fcb` |
| `crane_symeood_k1_candidate_selection_a_v1.py` | 已退出当前路线的候选排序控制；旧专项preflight使用时需从Git恢复。 | `72d7aa1b31ec831231d320929a24e89b3dfe073b1edd8cc4e9e40b4fc08eb302` |
| `crane_symeood_k1_candidate_selection_b_v1.py` | 已退出当前路线的候选排序监督；与控制一起恢复后才可运行旧专项preflight。 | `3a055e0f273838e67582dbd47aee8a61a8a4b230e71ea344f9c9755caa941b08` |
| `crane_symeood_k1_port_day2night_aug_c_v1.py` | 未纳入当前检测方法的旧光照增强。 | `54f15b60fcad423092256aa021324bb5de21615a2a86e7b05148cc73f3876e62` |
| `crane_symeood_mal_v1.py` | 已结束MAL实验的训练入口；旧MAL队列运行前需恢复该组配置。 | `b5775a3444a5615d0e8e5e14a0c784a755c754dd3c91775bb7093741b1c361f5` |
| `crane_symeood_mal_v1_base.py` | 已结束MAL/S_MAL公共配置，仅由本轮同时删除的两个入口继承。 | `56d0796944da8dea8edd5535dc9a8b877415b486fe2ea3c61fd3016b66c07766` |
| `crane_symeood_smal_v1.py` | 已按冻结数值保护停止的S_MAL入口，当前没有恢复训练安排。 | `dbc064af6d7b866cd2199f0dbf3d9e51a4757d9839332ffe760b84160a4fedc4` |
| `_base_/datasets/crane_dota.py` | 旧/root/mmrotate路径数据模板；现存配置无继承或有效引用。 | `eb5068f018c4e0a3deb2fa2189bd527bc64910d97d67774dd5a068d79e6a6a18` |

查看旧内容（不更改工作区）：

```bash
git show 030395be620d743b64db3bb94bb9ae8d8eb05828:crane_project/configs/crane_symeood_mal_v1.py
```

恢复所需文件（在项目根目录执行；先确认该路径没有需要保留的新改动）：

```bash
git restore --source=030395be620d743b64db3bb94bb9ae8d8eb05828 -- crane_project/configs/crane_symeood_mal_v1.py crane_project/configs/crane_symeood_mal_v1_base.py crane_project/configs/crane_symeood_smal_v1.py
```

旧候选专项脚本与历史文档中的原配置路径保持其历史含义；若将来确需重用，先从上述提交恢复相应配置和继承闭包，不能直接启动缺配置的旧队列。本次没有修改这些历史脚本或伪造新的配置。

Git历史不删除、不重写，也不额外复制一套旧配置。整理改动尚未自动提交或推送；`git status`可查看本次删除。实验原始配置快照、权重、日志、预测与既有结果均不改动。
