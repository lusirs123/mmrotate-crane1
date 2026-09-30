# DINO → SymEOOD 蒸馏实验总记录

本文件汇集 V1–V5 实验、同口径比较、warmstart V2 和迁移快照。前半部分按原 V3–V5 实验时间线记录；文末「合并的历史记录」保留各阶段原始表述，历史的“当前”与“下一步”不替代较新的裁决。

## V3–V5 FPN 梯度与关系蒸馏实验

本实验只检验原有前景特征损失能否在共享 FPN 获得蒸馏梯度后产生额外收益。V1/V2 配置、权重和结果保持原状。三个新训练均从同一 source VAL 选中的普通 K1 `epoch_20.pth` 初始化；不按已暴露 TEST 调损失或训练预算。

| 项目 | A | B | C |
| --- | --- | --- | --- |
| K1 初始化 | 同一 `epoch_20.pth` | 同左 | 同左 |
| ResNet-50 主干 | `frozen_stages=4` | 同左 | 同左 |
| FPN、检测头、分类适配器 | 可训练 | 可训练 | 可训练 |
| 蒸馏损失 | 无 | V2 前景余弦损失 | 同 B |
| 蒸馏梯度进入 FPN | 无 | `protect_geometry=True`，阻断 | `protect_geometry=False`，允许 |

三组均复用 V2 的 source `train + train_sim`、DINO 特征缓存管线、4 epoch、SGD 学习率 0.00025、每卡 batch 2 和 source VAL 选权规则。缓存读取在 A 中仍保留，以匹配数据路径和执行条件；A 不构建蒸馏投影器。B/C 只在 `protect_geometry` 上不同。主干冻结对三组一致，因此 V2 的旧结果只是历史参考，不能充当本实验的 B 组。

## 服务器运行

先将四个 `crane_symeood_k1_dino_fpn_gradient_*_v3.py` 配置和 `preflight_k1_dino_fpn_gradient_v3.py` 同步到服务器主仓库。以下命令从 `/media/omnisky/personal_files/ljj/symEOOD` 执行，使用 Python 3.8 的 `mmrotljj` 环境。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
CUDA_VISIBLE_DEVICES=0 python crane_project/tools/preflight_k1_dino_fpn_gradient_v3.py \
  --config-a crane_project/configs/crane_symeood_k1_dino_fpn_gradient_a_v3.py \
  --config-b crane_project/configs/crane_symeood_k1_dino_fpn_gradient_b_v3.py \
  --config-c crane_project/configs/crane_symeood_k1_dino_fpn_gradient_c_v3.py \
  --gpu 0 \
  --out-json work_dirs/crane_symeood_k1_dino_fpn_gradient_v3_preflight.json
```

只有预检输出 `MATCHED_FPN_GRADIENT_READY` 时才依次训练。预检使用一张 source 图像和已存在的缓存，不保存权重，也不读取 TEST。它检查 K1 权重 SHA、配置差异、实际参数冻结、前景掩码、蒸馏专属梯度、总训练梯度、一次优化后的分类输出变化和峰值显存。预检的单步更新只发生在临时内存中的模型。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh crane_project/configs/crane_symeood_k1_dino_fpn_gradient_a_v3.py 2 --no-validate
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh crane_project/configs/crane_symeood_k1_dino_fpn_gradient_b_v3.py 2 --no-validate
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh crane_project/configs/crane_symeood_k1_dino_fpn_gradient_c_v3.py 2 --no-validate
```

三条训练命令应顺序执行，确认上一条成功后再执行下一条；脚本固定 `--seed 0`。`--no-validate` 关闭训练过程中的在线 VAL，仍按配置每 epoch 保存权重；之后独立按原协议扫描 source VAL。先检查三个训练目录中是否各有 `epoch_1.pth` 至 `epoch_4.pth`，以及日志中实际学习率、batch、参数冻结和损失。

选权推理统一使用 `crane_symeood_k1_dino_semantic_student_v1.py`：它保留训练得到的分类适配器，不向推理模型传入训练期 `semantic_distillation` 参数。2026-09-24 修正了 `ckpt_sweep.py` 调用 `tools/test.py` 时缺少仓库根目录 `PYTHONPATH` 的问题；服务器必须同步该脚本后再执行下列 sweep。先前 A/epoch_1 的失败发生在模型构建阶段，没有产生可用 VAL 结果；新 sweep 目录与失败目录隔离，保留其日志。

```bash
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py \
  --work-dir work_dirs/crane_symeood_k1_dino_fpn_gradient_a_v3 \
  --sweep-dir work_dirs/crane_symeood_k1_dino_fpn_gradient_a_v3/source_val_sweep_student_protocol_v2 \
  --epochs 1 2 3 4 --gpu 0
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py \
  --work-dir work_dirs/crane_symeood_k1_dino_fpn_gradient_b_v3 \
  --sweep-dir work_dirs/crane_symeood_k1_dino_fpn_gradient_b_v3/source_val_sweep_student_protocol_v2 \
  --epochs 1 2 3 4 --gpu 0
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py \
  --work-dir work_dirs/crane_symeood_k1_dino_fpn_gradient_c_v3 \
  --sweep-dir work_dirs/crane_symeood_k1_dino_fpn_gradient_c_v3/source_val_sweep_student_protocol_v2 \
  --epochs 1 2 3 4 --gpu 0
```

这些 sweep 命令只扫描 source VAL，不带 `--run-final-test`。先报告三组 source VAL 选权、分类输出覆盖、全帧中心命中、RIoU、角度和 MCML，再决定是否进行一次冻结 TEST。比较 C−A 衡量有无蒸馏额外收益；比较 C−B 衡量 FPN 梯度路由的作用。主干即使冻结，FPN 更新仍会改变回归分支输入，不能预设几何不退化。

本地缺少 MMCV/MMRotate，不能完成真实模型构建、GPU 梯度及缓存验证。服务器预检通过之前，不将表中的运行期状态视为已验证。

## 2026-09-24：既有 A/B/C 结果核查与修复

三组固定 TEST 均选中 `epoch_1`，每组导出 877 个框。real 全帧中心命中率均为 70.71%，real 最长连续缺测均为 38；DFR 等浮点指标末位略有不同。这些汇总不足以判断三组逐帧预测是否相同。以下核查只读取已生成的 VAL、TEST、checkpoint 和日志，不重新推理、不选权。

代码审查发现原前景掩码忽略 OBB 角度，用未旋转的 `w,h` 计算区域；现已用旋转框真实轴向外接宽高计算。原损失用最近邻把 FPN 前景掩码缩到教师网格时，单格小目标可能消失；现改为缩小时进行最大池化。两项修复影响**未来训练**，不会改变已完成的 A/B/C checkpoint，不能把旧结果解释成使用了修复后的训练实现。

上方 V3 训练命令是历史运行记录。修复后若开展新训练，应使用新的实验名称和输出目录，避免同名 V3 混入不同损失实现；本次核查不启动训练。

原 `ckpt_sweep.py` 只要看见既有 `results.pkl` 就跳过推理，未核对生成它的配置和权重；DOTA 导出缓存也只核对帧名。现在新推理完成后写入 `results.pkl.provenance.json`，包含配置、checkpoint、标注和 PKL 哈希；新 DOTA 导出记录源 PKL 与导出文件哈希。复用时要求这些记录完整匹配。历史 PKL 若没有生成记录，核查脚本会标记为 `historical_generation_unattested`，即使报告中保存的 checkpoint 和 PKL 哈希一致，也不把它误称为已证明生成关系。不要为旧 PKL 手工补写此记录。

将本地改动同步到服务器后，在项目根目录运行一次只读核查。脚本只额外写出一份 JSON 报告；若路径、哈希或逐帧 DOTA/PKL 几何不一致，会直接报错。它还比较三组学生与 K1 初始化权重，核对历史梯度预检（若存在），逐帧比较 A/B/C 的输出、中心命中和框/分数差异，并汇总训练日志中的蒸馏损失记录。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/audit_k1_dino_fpn_gradient_v3.py \
  --project-root . \
  --out-json work_dirs/crane_symeood_k1_dino_fpn_gradient_v3_artifact_audit.json
```

读报告时优先看 `weight_comparisons` 的 backbone/FPN/adapter/head 差异、`paired_predictions` 中 real/sim 的独有输出和独有中心命中，以及各组的 `training_log`。`historical_generation_unattested` 是证据边界，不等于已经串用预测。核查脚本无法从旧 PKL 的内容反推出当时使用的 checkpoint；若权重和逐帧预测都正常，三组指标相似才可解释为当前损失与预算下效果很小。

下载核查结果时可在服务器根目录打包已有报告与三组原始 TEST 预测，不包含体积较大的权重：

```bash
tar -czf dino_fpn_v3_audit_results_20260924.tar.gz \
  work_dirs/crane_symeood_k1_dino_fpn_gradient_v3_artifact_audit.json \
  work_dirs/crane_symeood_k1_dino_fpn_gradient_{a,b,c}_v3/source_val_sweep_student_protocol_v2/sweep_results.json \
  work_dirs/crane_symeood_k1_dino_fpn_gradient_{a,b,c}_v3/source_val_sweep_student_protocol_v2/final_test/epoch_1/final_test_metrics_v2.json \
  work_dirs/crane_symeood_k1_dino_fpn_gradient_{a,b,c}_v3/source_val_sweep_student_protocol_v2/final_test/epoch_1/preds/results.pkl
```

这份 TEST 已暴露，核查只作诊断。若确认学生参数和预测真实不同，但任务结果几乎不变，再在下一次预定 source 实验中测量蒸馏梯度相对检测梯度的大小，并用修复后的掩码做受控对照。

## 2026-09-25：V3 结论固定与修正掩码 source 探针

已归档的服务器产物核查报告、历史预检、三组训练日志和 TEST PKL 表明：A/B/C 均由 K1 `epoch_20.pth` 初始化，主干状态与 K1 完全相同；A 无蒸馏损失，B 的蒸馏梯度停在分类适配器，C 的蒸馏梯度到达 FPN。三组选权均为 `epoch_1`，四个候选都满足 source VAL 约束，第一轮软评分最高。B/C 的权重与 A 不同；C 的 FPN 与 A 的 L2 差异约 0.07259，B 与 A 仅约 0.0000476。完整训练中蒸馏损失 B 约从 0.05198 降至 0.04810，C 约从 0.05196 降至 0.03938。

固定 TEST 的 992 帧上，三组在完全相同的 877 帧输出：real 305 帧输出、115 帧无输出，sim 572 帧全有输出；中心命中帧集合完全相同。real `R_center=70.71%`、`mean_RIoU=0.5003`、`MCML_max=38` 三组相同。原始 PKL 不相同，C 相对 A 的中心坐标最大差约 0.06 像素、分数最大差约 0.00072。**V3 结论：开放 FPN 蒸馏梯度改善了训练特征对齐，却没有改变本次选中权重的输出覆盖、中心命中或最长连续 RIoU 失败。** 这不能推出 DINO 任务知识不可迁移。历史 PKL 缺少生成时的 provenance sidecar，checkpoint→PKL 的生成关系仍只由当时目录和报告支持，不把它表述为严格证明。

完整 C 训练有一次仅到 iter 100 的启动日志和一次完整四轮日志；不能把两份日志计数相加当作额外训练预算。审计报告中检测头相对 K1 的约 3480 最大差值包含训练计数缓冲区，不代表检测卷积权重大幅变化。原 V3 训练使用未考虑 OBB 角度的前景掩码和最近邻缩小；后续代码修正不会追溯改变 V3 权重。

下一步只运行一次短 source 探针 `probe_k1_dino_corrected_mask_source_v4.py`。它用原 A/B/C 配置核对身份，但只构建 C 学生，从同一 K1 权重出发；固定读取 source `train` 与 `train_sim` 的首帧和现有 DINO 缓存。运行前先用合成旋转框和单格前景验证服务器实际加载了修正后的掩码与缩小逻辑，并记录两份实现文件的 SHA。它报告修正前后掩码的 FPN 像素与教师 token、检测与蒸馏损失分别作用于 FPN 的梯度范数/余弦，以及一次**仅在内存中**按配置学习率沿蒸馏梯度更新 FPN 与分类适配器后的分类 logit 和概率变化。这一步不使用动量、权重衰减或梯度裁剪，只衡量局部响应，不等价于真实训练预算下的更新，也不用于选 checkpoint 或调整 TEST 阈值。若单步响应低于 float32 分辨率，报告零值而不编造收益。

同步本地脚本及修正后的掩码/损失实现后，在服务器仓库根目录执行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python \
  crane_project/tools/probe_k1_dino_corrected_mask_source_v4.py \
  --project-root . --gpu 0 \
  --out-json work_dirs/crane_symeood_k1_dino_corrected_mask_source_probe_v4.json
```

脚本拒绝覆盖既有 JSON。核对报告中的 `corrected_teacher_tokens`、`fpn_gradients`（尤其范数比和余弦）及 `temporary_distillation_step_response`，先判断修正后的监督是否在 source 样本上影响学生用于检测的表示与分类分数。只采样两帧，结果是设计诊断，不能据此声明总体检测性能提升。后续若需正式实验，应另起新编号、保留同预算无蒸馏对照，并只用 source VAL 选权。

## 2026-09-25：修正掩码后的 A/C 同预算对照 V4

短 source 探针的服务器结果使用同一 K1 `epoch_20.pth`（SHA256 `3ab0885159294beb820da1445c38045a342fd4956c3d094eeaaadf78deb745c2`），运行时证实旋转框掩码与最大池化缩小已生效。两帧合并后，蒸馏梯度确实进入 FPN：蒸馏/检测梯度范数比约 `0.01444`，余弦约 `0.01460`；一次仅沿蒸馏梯度的临时更新改变了 P3 与分类输出。该探针只有两帧、零训练轮，不能证明检测收益，也不支持按梯度范数比直接放大蒸馏权重。

本次只训练两个新组，专门验证**修正后的原特征损失**在相同预算下是否带来任务收益。A/C 均继承 V3 公共配置：同一 K1 初始化，冻结 ResNet-50，FPN、检测头与分类适配器可训练；同一 source `train + train_sim` 和 DINO 缓存管线；4 epoch、每卡 batch 2、SGD `lr=0.00025`、相同学习率计划、梯度裁剪与 seed 0。A 无蒸馏损失；C 使用原前景特征余弦损失 `loss_weight=0.05`，且 `protect_geometry=False`。两组只在蒸馏设置及各自输出目录不同。掩码修复位于模型和损失实现中，旧 V3 权重不属于此实验。B 不重训，因为当前要回答的是修正后 C 相对无蒸馏 A 的额外收益。

同步两份新配置、`preflight_k1_dino_corrected_mask_ac_v4.py` 和当前掩码/损失实现到服务器后，在仓库根目录依次执行。预检使用一张 source 图像对 A/C 各做一次临时优化，检查 K1 SHA、完整配置一致性、掩码修复运行时、参数冻结、A 无蒸馏损失、C 的蒸馏梯度进入 FPN、同图同缓存及输出变化；不保存训练权重。`CORRECTED_MASK_AC_READY` 出现后再训练。新目录应为空，避免混入旧产物。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
test ! -e work_dirs/crane_symeood_k1_dino_corrected_mask_a_v4
test ! -e work_dirs/crane_symeood_k1_dino_corrected_mask_c_v4
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python \
  crane_project/tools/preflight_k1_dino_corrected_mask_ac_v4.py \
  --config-a crane_project/configs/crane_symeood_k1_dino_corrected_mask_a_v4.py \
  --config-c crane_project/configs/crane_symeood_k1_dino_corrected_mask_c_v4.py \
  --gpu 0 \
  --out-json work_dirs/crane_symeood_k1_dino_corrected_mask_ac_v4_preflight.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_dino_corrected_mask_a_v4.py 2 --no-validate
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_dino_corrected_mask_c_v4.py 2 --no-validate
```

确认每组都有 `epoch_1.pth` 至 `epoch_4.pth`，且训练日志无 NaN、学习率和轮数符合配置，然后逐组独立扫描 source VAL。推理统一使用保留分类适配器的学生配置，训练期投影器不会进入推理模型；`ckpt_sweep.py` 会为新预测写入 provenance。此阶段不传 `--run-final-test`。

```bash
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py \
  --work-dir work_dirs/crane_symeood_k1_dino_corrected_mask_a_v4 \
  --sweep-dir work_dirs/crane_symeood_k1_dino_corrected_mask_a_v4/source_val_sweep_protocol_v2 \
  --epochs 1 2 3 4 --gpu 0
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py \
  --work-dir work_dirs/crane_symeood_k1_dino_corrected_mask_c_v4 \
  --sweep-dir work_dirs/crane_symeood_k1_dino_corrected_mask_c_v4/source_val_sweep_protocol_v2 \
  --epochs 1 2 3 4 --gpu 0
```

分析时先比较 source VAL 选中权重的同帧预测：C 新增和丢失的正确输出、几何精度、连续 RIoU 失败，以及蒸馏损失下降是否转为任务收益。两组各自按相同 source VAL 规则选权。只有在 source 结论固定后，才决定是否做一次固定 TEST；不要按既有 TEST 的单帧差异调损失。

如需把 source 阶段产物下载到本地，可在服务器仓库根目录打包两组结果和预检。命令排除 checkpoint，但保留日志、source VAL 指标、原始预测及其 provenance，便于逐帧复查：

```bash
tar -czf dino_corrected_mask_ac_v4_source_results.tar.gz \
  --exclude='*.pth' \
  work_dirs/crane_symeood_k1_dino_corrected_mask_ac_v4_preflight.json \
  work_dirs/crane_symeood_k1_dino_corrected_mask_a_v4 \
  work_dirs/crane_symeood_k1_dino_corrected_mask_c_v4
```

## 2026-09-25：对象—邻近背景关系蒸馏的短 source 检查 V5

V4 的 A/C 均在 source VAL 选中 epoch 1；固定 TEST 上 real 均输出 305/420 帧，其中 297 帧中心误差小于 15 px，输出帧中心命中率均为 297/305=97.38%，最长连续 RIoU 失败均为 38 帧。C 的特征损失下降，但输出帧集合、中心命中状态和 RIoU 达标状态与 A 完全相同。本项目后续把“中心命中率”用于**有输出帧**的条件定位评价，并同时单列输出覆盖率、连续失败；旧报告中的全帧 `R_center` 保留原标签及分母，不能直接与条件命中率相减。

本探针只检查下一种监督是否值得进入受控实验，核心问题是：**教师的对象—邻近背景关系是否具有可留出的区分信息，学生分类适配器能否在匹配的短更新中学习该关系，并改善对象相对背景的分类响应，同时保留原有检测输出。** 它不把非零梯度或同图损失下降单独当作可行性结论。

`probe_k1_dino_object_background_source_v5.py` 复用现有 K1 `epoch_20.pth` 和 DINO 缓存，按标注预先选择 fit 与 held-out 图像：real 使用交替的完整序列分离，sim 当前只有 `sim_seq08`，因此使用按帧号前后半段的明确 fallback，并在报告中标记 `single_sequence_frame_block_split`。每个域各取 fit 两张、held-out 两张，按 GT 短边的 25%/75% 分位选取。选择不读取预测。sim 的 fallback 不是序列独立泛化证据，只用于检查关系监督链路是否能学习。每帧只接受一个 GT 对象；旋转框内部作为对象区域，在一格间隔外取近邻背景环，并排除 padding。教师区分度用对象 token 的交替留出子集构造原型，报告 held-out 对象相对背景的 gap/AUC；对象 token 不足时记为不可用。

探针包含两个完全匹配的短更新分支：

* `control`：K1 主检测损失；
* `relation`：同一主检测损失，加 `0.05 ×` 对象/背景等权关系 MSE。

两组都从同一个 K1 权重开始，只让分类适配器可训练，FPN、回归分支和其余参数冻结；每组 10 步，使用原 K1 SGD 的动量、权重衰减和梯度裁剪，固定使用原 K1 基础学习率 `0.0025`，不使用 warmup。这是短的设计探针，不等价于 K1 的 24 epoch 训练，也不保存 checkpoint。每一步在 real/sim fit 图像间交替，held-out 图像只用于评估。脚本恢复适配器、检测损失计数器和缓冲区后再运行另一分支。

判据预先写在脚本中，属于小样本探索性门槛，不是统计显著性检验。继续做受控训练至少需要两个域都满足：教师 held-out AUC≥0.6 且 gap≥0.02；relation 分支相对 control 的 held-out 关系 MSE 至少改善 1%；对象相对背景的分类 score gap 有正增量；学生 held-out AUC/gap 不低于 control；对象 score 不下降；输出、15 px 中心命中和 RIoU 命中均没有丢失。任何教师 AUC≤0.5、检测输出或命中丢失、或分类 gap 下降，记为本探针未支持。其余情况记为证据不足。`automatic_training_authorized` 永远为 `false`。此外，脚本记录关系损失到适配器的独立梯度范数，并对 baseline/control/relation 的回归分支做 SHA256 一致性检查，防止把回归改变误归因于分类关系监督。

服务器在仓库根目录同步新探针后运行以下单条命令：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python \
  crane_project/tools/probe_k1_dino_object_background_source_v5.py \
  --project-root . --gpu 0 \
  --out-json work_dirs/crane_symeood_k1_dino_object_background_source_probe_v5_r2.json
```

脚本拒绝覆盖已有 JSON。重点查看 `assessment.status`、每个域的 `teacher_auc`/`teacher_gap`、`relative_mse_gain_over_control`、`student_auc_gain_over_control`、`score_gap_gain_over_control`、`lost_outputs`/`lost_center_hits`/`lost_riou_hits`，以及首步 `relation_adapter_gradient_norm`。不要把 `supports_next_controlled_experiment` 解释成正式训练已获批准；它只表示这组短 source 证据达到预先门槛。探针不读取 VAL/TEST，不选择 checkpoint，不调整 TEST 阈值。若证据不足或未支持，先记录限制，不重新做全量教师诊断。

如需把报告下载到本地，可在服务器仓库根目录打包：

```bash
tar -czf dino_object_background_source_probe_v5_r2.tar.gz \
  work_dirs/crane_symeood_k1_dino_object_background_source_probe_v5_r2.json
```

## 2026-09-26：对象—邻近背景关系蒸馏正式 A/C 训练 V5

短探针只验证了教师关系存在、关系梯度能到达分类适配器；它没有运行完整训练，因此不能作为正式效果结论。正式实验接入新的 `ObjectBackgroundRelationDistillation` 训练损失，不复用短探针作为训练脚本。

配置为 A/C 两组。两组都从 K1 `epoch_20.pth` 初始化，使用 real train + sim train、batch 2、`frozen_stages=1`、SGD `lr=0.0025`、momentum `0.9`、weight decay `0.0001`、梯度裁剪 10、warmup 1000、step `[16,22]`，运行 24 epoch。C 的关系损失必须读取离线 DINO `teacher_features`；A 也读取同一缓存但忽略它，以保持数据管线和 batch 输入完全匹配，避免把缓存读取差异混入 A/C 对照。FPN、检测头和分类适配器由原检测损失训练。A 不启用新关系损失；C 只增加 `0.05 ×` 对象/邻近背景关系 MSE。关系项保护 FPN 几何路径，只通过分类适配器更新；两组的主检测损失仍共同更新 FPN 和检测头。

正式损失从变换后的 GT 旋转框生成精确对象区域，在对象外一格到四格的邻近环中取背景，并排除 padding。教师和学生分别用对象特征原型计算各位置的余弦关系，对象与背景 MSE 等权平均。教师特征来自现有离线 DINO cache，推理阶段不读取教师特征，也不改变回归分支。

本地已完成关系损失模块、A/C 配置和配置预检脚本的语法检查；本机无 MMCV/CUDA，不能代替服务器预检。同步以下文件：`mmrotate/models/losses/object_background_relation.py`、`mmrotate/models/detectors/sym_eood_detector.py`、`crane_project/configs/crane_symeood_k1_dino_object_background_relation_common_v5.py`、`crane_project/configs/crane_symeood_k1_dino_object_background_relation_a_v5.py`、`crane_project/configs/crane_symeood_k1_dino_object_background_relation_c_v5.py`、`crane_project/tools/preflight_k1_dino_object_background_relation_v5.py`、`crane_project/tools/audit_k1_dino_object_background_relation_v5.py` 和 `mmrotate/models/losses/__init__.py`。

服务器先做配置预检：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
test ! -e work_dirs/crane_symeood_k1_dino_object_background_relation_a_v5
test ! -e work_dirs/crane_symeood_k1_dino_object_background_relation_c_v5
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/preflight_k1_dino_object_background_relation_v5.py \
  --config-a crane_project/configs/crane_symeood_k1_dino_object_background_relation_a_v5.py \
  --config-c crane_project/configs/crane_symeood_k1_dino_object_background_relation_c_v5.py \
  --out-json work_dirs/crane_symeood_k1_dino_object_background_relation_v5_preflight.json
```

只在输出 `READY_FOR_FORMAL_TRAINING` 后训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_dino_object_background_relation_a_v5.py 2 --no-validate
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_dino_object_background_relation_c_v5.py 2 --no-validate
```

训练完成后，先逐组扫描 source VAL，再根据同一规则选 epoch。推理配置只需保留分类适配器，不启用训练期关系损失：

```bash
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py \
  --work-dir work_dirs/crane_symeood_k1_dino_object_background_relation_a_v5 \
  --sweep-dir work_dirs/crane_symeood_k1_dino_object_background_relation_a_v5/source_val_sweep_protocol_v2 \
  --epochs 16 18 20 22 24 --gpu 3
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py \
  --work-dir work_dirs/crane_symeood_k1_dino_object_background_relation_c_v5 \
  --sweep-dir work_dirs/crane_symeood_k1_dino_object_background_relation_c_v5/source_val_sweep_protocol_v2 \
  --epochs 16 18 20 22 24 --gpu 3
```

分析时优先比较 A/C 的新增和丢失正确输出、输出覆盖率、仅有输出帧的中心命中率、RIoU 和连续失败长度。关系损失下降不能单独作为成功标准。当前不根据固定 TEST 调损失；只有 source VAL 结论固定后才决定是否进行一次 TEST。

### V5 固定 TEST 结论与产物核查（2026-09-26）

source VAL 选权已经固定：A 使用 `epoch_24`，C 使用 `epoch_18`。两组随后在固定 992 帧 TEST 上完成评估。real 域的结果为：A 的 `R_center=60.24%`、`mean_RIoU=0.4433`、`TDR_w10=67.43%`、`MCML_max=64`、`MCML_mean=34.67`、`MRF=16.45`；C 的 `R_center=60.24%`、`mean_RIoU=0.4460`、`TDR_w10=69.47%`、`MCML_max=64`、`MCML_mean=33.67`、`MRF=10.53`。sim 域 A/C 均为 `R_center=100%`、`TDR_w10=100%`、`MCML_max=0`，C 的 `mean_RIoU=0.8724` 低于 A 的 `0.8871`，角度误差也由 `1.3104°` 变为 `1.7880°`。

因此本轮固定结论是：C 在 real 域的部分时序统计量有小幅改善，但没有改善中心命中率或最长连续缺测，sim 域略有退化。当前证据不足以支持对象—邻近背景关系蒸馏作为有效改进方案；不根据这组 TEST 结果继续调损失权重、背景环或训练轮数。`CraneOfflineEvaluator [TEST 模式]` 是评估器的完整时序指标模式名称，以上结果实际来自固定 TEST，而不是 source VAL 重跑。

服务器上可对既有产物进行只读核查。该脚本不运行推理、不重新选权、不写入 provenance sidecar；它核对 A/C 的正式训练配置（K1 `epoch_20.pth` 初始化、24 epoch、冻结主干、关系字段、DINO cache 和 `teacher_features`）、source-VAL 选权记录、checkpoint/config/PKL 哈希、固定 TEST 报告身份、预测 provenance，以及训练日志中 C 的非零关系损失和 A 的无关系损失记录。多个训练日志文件只记录为 warning，不自动把重复运行判为模型错误：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/audit_k1_dino_object_background_relation_v5.py \
  --project-root . \
  --out-json work_dirs/crane_symeood_k1_dino_object_background_relation_v5_artifact_audit_complete.json
```

核查报告中的 `c_minus_a_metrics` 用于记录 TEST 差值，`conclusion` 固定为本轮“real 时序有弱变化、中心命中率和 `MCML_max` 无改善”的证据边界。若核查失败，应先修复产物身份或日志问题，不重新解释模型效果。


### 2026-09-27：审计范围纠正与一次性完成核查

此前报告的 `provenance=verified` 只覆盖部分字段，`conclusion` 为硬编码，不能把它当作自动验证的实验结论。历史 9 个测试针对关系损失和短探针，不构成审计脚本覆盖证明。本次新增专门的审计测试。

当前审计在既有产物上核对全部五个 source VAL checkpoint/PKL/sidecar、VAL/TEST 标注哈希、DOTA 导出哈希与帧集合、固定 TEST 指标重算、PKL 与 DOTA 几何对应。读取 checkpoint 的 epoch、适配器权重和历史 config（若存在），对照当前解析配置；日志分别保留每个文件的计数与哈希。缺少历史配置或无法绑定 A 的两份日志时显式保留限制，不能据此宣布完全追溯通过。适配器非零和正损失也不能单独证明关系损失的梯度贡献。

更正指标解释：当前历史评估器 `R_center` 分母包含无输出的 GT 帧，之前将 60.24% 当作仅输出帧命中率的解释不正确。新审计 `paired_test.groups.*.methods.*.center_hit_given_output` 为用户要求的条件中心命中率，同时输出 `output_coverage`。历史 `MCML` 是 RIoU<0.5 的连续失败，不能等同于连续无输出。新报告分别列出 output/center_hit/riou_hit 的失败区间，并在序列及帧号断点处分段。保留旧选权和旧报告以维护证据身份。

运行（CPU，不重跑网络推理）：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/audit_k1_dino_object_background_relation_v5.py \
  --project-root . \
  --out-json work_dirs/crane_symeood_k1_dino_object_background_relation_v5_artifact_audit_complete.json
```

核查输出包含逐帧新增/丢失中心命中、覆盖率、几何精度及三类失败区间。历史指标支持“未见明确整体收益”，条件命中率和最长无输出区间须以新核算为准；不继续将旧解释作为固定事实。

### V5 source VAL 候选流失诊断

此阶段只诊断已固定的 A `epoch_24` 和 C `epoch_18`，不训练、不改阈值、不读取 fixed TEST。目标为 source VAL 的 `real_seq07` 与 `sim_seq10` 共 738 帧。先检查已有 VAL `results.pkl`：它只保存最终最多一个框，无法还原阈值前候选，因此脚本对固定权重各做一次前向追踪，并将每帧追踪得到的 top-1 框与原 PKL 对齐。对齐失败时终止，不生成诊断报告。

此前提到的暗光 33 帧、远距 40 帧和小目标 64 帧属于 fixed TEST 的 `real_seq02/03` 诊断切片，不能称为本次 source VAL 的失败帧，也不能借 source VAL 的分层结果逐帧解释它们。本次分析只能帮助确定 source 域中哪一层最常流失候选；能否解释 target 域的 64 帧连续失败仍是后续泛化问题。

SymEOOD 当前推理是单阶段检测头：密集锚点解码 → padding 锚点过滤 → `score_thr=0.05` → 全局 `max_per_img=1`。没有 RPN、分类前 top-K 或 NMS；配置中的 `nms_pre` 与 `nms` 在该路径上未执行。诊断按实际路径记录每帧所有有效解码候选的最佳 RIoU、RIoU≥0.5 的候选数及最佳分数/排名、过阈值候选数、最终 top-1 的 RIoU，并把失败归为：无几何合格候选、分数阈值流失、top-1 排序流失。仅因某组多输出或少输出不能推断它改善了几何合格候选。报告按 real/sim 汇总并列出 A/C 输出集合变化和阶段转移。

服务器使用第四张卡运行。运行前先同步 `crane_project/tools/diagnose_v5_source_val_candidate_flow.py`，命令须从仓库根目录执行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/diagnose_v5_source_val_candidate_flow.py \
  --project-root . --gpu 3 \
  --out-json work_dirs/crane_symeood_k1_dino_object_background_relation_v5_source_val_candidate_flow_v1.json
```

脚本核对 VAL 标注、选权文件、checkpoint/PKL 哈希与预测 provenance；只读 source VAL，输出协议为 `v5_source_val_candidate_flow_v1` 的 JSON。RIoU≥0.5 是本次几何候选诊断阈值，仅用于分层；候选覆盖率低不能单独证明 FPN 容量或锚点设计错误。先看阶段分布、A/C 新增/丢失帧的阶段，以及原始 top-1 一致性，再确定是否需要下一项实验。

### 历史机制证据清单

在决定下一项分类或质量引导实验前，先在完整服务器 checkout 执行只读清单脚本。它扫描配置、代码、文档、日志、JSON 和实验目录名，分别标记实现/配置证据、历史记录证据和测试/引用文本；不会构建模型、运行推理、训练或使用 TEST 指标选方向。`historical_record_present_review_experiment_identity` 只表示找到历史记录，还需把配置、checkpoint、日志和 source VAL 结果绑定后才能确认“已验证”。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/audit_historical_mechanisms_v1.py \
  --project-root . \
  --out-json work_dirs/historical_mechanism_inventory_final.json
```

优先查看 JSON 中 `assignment`、`classification`、`quality_guidance` 三组的 `status`、`bound_artifacts` 和 `evidence_class_counts`。只有 `historical_artifact_bound_review_scope_and_result` 才进入实验候选核查；`historical_text_only_no_bound_artifact` 只是历史文字线索。脚本会排除自身和旧清单 JSON，并要求同一实验目录同时有配置、checkpoint、训练日志和 source VAL 证据。涉及 TEST 文件的条目只用于追溯，不用于选择阈值、checkpoint 或训练方向。将该清单与 27 条 S7 记录、V1–V5 训练日志和 source VAL 选权文件一起核对后，才决定是否开展单一 A/B 实验。

### 历史机制核查的最终口径（2026-09-27）

本次核查的目的，是防止下一项改进重复已经做过的机制。证据分三层：配置文件证明机制已经进入过实验流程；workdir 中的 checkpoint、训练日志和 source VAL 结果用于复核当时的结果；文档只补充设计背景、失败记录和实验限制。workdir 被清理时，仍保留“做过”的结论，但该实验的数值只能标为“结果不可复核”。TEST 文件只作为历史追溯，不参与下一项方案选择。

当前 K1 的身份已经固定：`crane_project/configs/crane_symeood_k1.py` 的主头使用 `SymPOLAAssigner` 和 `SymNFLLoss`，另有一个只参与训练的 `RotatedATSSHead` 辅助头；推理主路径使用 SymEOODHead。通用 `MaxIoUAssigner`、Rotated ATSS、FCOS centerness 和 RePoints/PAA 配置不能直接改写成 K1 主头实验。

基于仓库现有配置、代码、历史文档和已保存结果，机制清单如下：

| 机制 | 是否进入过项目实验配置 | 当前能否复核结果 | 后续处理 |
| --- | --- | --- | --- |
| SymPOLA + SymNFL | 是，当前 K1 主路径 | K1/V1–V5 结果按各自产物记录 | 作为基线，不重复更换名称 |
| ATSS | 是，K1 辅助头及其他旋转检测器 | 需按模型身份区分 | 不作为新的 K1 主分配实验 |
| MaxIoU | 是，EOOD/通用检测器 | 与 K1 主头不等价 | 不直接迁移结论 |
| PAA/RePoints | 有通用实现或配置 | 未确认 K1 主头结果 | 不把代码存在当作 K1 结果 |
| SimOTA | 有文字或引用线索 | 未找到明确 K1 结果绑定 | 若 config 存在则归为已做，未有 config 则只记线索 |
| 质量下界 | 有讨论和部分质量阈值文字 | 未确认独立 K1 消融 | 先核对现有 SymPOLA 代码，避免重复 |
| QFL | 历史文档记录已试且退化 | 原始产物需单独绑定 | 作为已有负结果，不重复 |
| VFL | 文档记录未形成已验证结果 | 没有可靠 K1 产物 | 不因“未试”就直接训练，先判断与 QFL 的新增信息 |
| IoU-aware head | 有设计和文献映射 | 未确认 K1 训练结果 | 不能称已验证方案 |
| centerness | 通用 FCOS/ATSS 配置存在 | 非 K1 主头结果 | 不直接迁移 |
| RegQuality/PQA | 文档记录 v1–v3 失败或封存 | 结果产物需绑定 | 不换名称重做 |
| score modulation | 检测器代码存在若干质量/分数接口 | 未形成当前 K1 的独立结果 | 不能仅凭代码存在宣称有效 |
| DINO S7/quality ranking | 有独立 source-only 训练和诊断记录 | 属于 DINO RPN/ROI 候选链 | 与 K1 蒸馏分开解释 |

这份清单支持的研究结论是：V5 候选流失诊断暴露的是分类分数与几何质量脱钩；V5 对象—邻近背景关系蒸馏没有形成整体收益；QFL、RegQuality/PQA 和多条 DINO/S7 排序路线已有负结果或边界限制。下一项工作不能重复这些路线，也不能因为某个通用模块存在就宣称它已经在 K1 上验证。

如果继续实验，只允许在已有配置和结果核对完成后选择一个明确新增的机制。优先检查候选级相对排序是否已经在 **K1 主头** 做过；DINO S7 的 relative-quality 结果不能代替 K1 实验。若 K1 中确实没有该机制，才设计单一 source-only A/B 对照，并预先固定旧正确帧保持、成功帧、覆盖率、RIoU、MCML 和连续无输出长度的判据。

### K1 主头候选排序静态核对（2026-09-27）

范围：本地保留的 K1/DINO 配置、SymEOOD 主头及 detector 训练调用、历史排序实验文档。配置未找到不等于历史没有做过；按项目约定，项目实验配置存在即可记为做过，work_dir 删除不撤销这一身份。通用上游配置不作为本项目运行证据。

| 检查项 | 当前代码/配置事实 | 历史边界 |
| --- | --- | --- |
| K1 主头分类与分配 | `crane_symeood_k1.py` 使用 SymNFL（gamma=2、alpha=0.25）和 SymPOLA；ATSS 用于辅助头 | 不能把辅助头 ATSS 当作主头分配消融 |
| 已有几何监督 | SymNFL 用 detached SymKLD 加权分类损失；SymPOLA 联合分类与几何代价 | K1 已有几何信息，不能以“首次引入几何质量”为新实验理由 |
| QFL 分支 | `sym_nfl_loss.py` 有 `use_quality_target`，默认关闭，当前 K1 未开启 | 历史 QFL 退化记录保留，不因当前关闭就记为未做 |
| 独立质量头 | 当前 K1 配置未启用 `reg_quality_head`、`pqa_head` | RegQuality/PQA 已有失败或封存记录 |
| PQA pairwise 排序 | detector 已实现 `_build_pqa_rank_batches`、`_compute_pqa_rank_loss`，对 PQA 热图采样得到的 quality 用候选 IoU 做成对排序；权重默认 0 | 本地未找到启用该具体 rank 开关的配置，不能据此断言未试；“候选级相对排序”这个名称本身已不足以区分新旧方案 |
| DINO/S7 排序 | 文档记载 Pairwise V1/V2、relative-quality、unified hard-pair | 属于 DINO RPN/ROI 路线，结论不能直接移作 K1 主头结果 |
| 直接监督 K1 原始分类分数的候选成对损失 | 当前保留的 K1 配置及训练调用中未发现启用证据 | 仅为本次静态检索结果，不是“整个项目从未做过”的证明 |

结论：不能把下一步简化为“加一个 ranking loss”。可讨论的差异是**直接约束部署所用的 K1 分类分数**，与已有 PQA 独立质量输出、DINO ROI 排序区分；其有效性尚无证据。即使改变监督作用位置，也不能自动证明具有足够新增价值。

本轮静态核对到此结束，无需服务器重跑。若继续，先在方案中明确候选来源、IoU 成对目标、作用分数、梯度范围和旧正确输出保持方式，并逐项对照现有 PQA/S7 实现；只有这些差异足以形成独立假设，才讨论同预算 source-only A/B。尤其要处理低分但排序正确的情况：成对排序只约束相对次序，不能保证最高分超过固定 score threshold。当前诊断含 score_threshold 失败，因此单独 pairwise 不能覆盖全部问题。本轮不修改模型、不启动训练。

### 静态核对后的单一实验设计草案（2026-09-27，未实现/未训练）

**假设**：在存在 RIoU≥0.5 候选的 source 样本上，直接约束 K1 部署所用分类 logit 的最终选择条件，可能比独立质量头/特征关系监督更贴近输出失败。V5 A/C 的候选覆盖结果只能支持研究问题，不能替代 K1 epoch20 或新对照的实际表现，也不证明本方案有效。

**唯一新增项：包含固定输出阈值的候选选择损失。** 对每幅有 GT 的训练图像，按真实推理路径排除 padding 锚点；对全部有效候选解码，用 detached 框和 GT 算旋转 IoU（不先按分数截断候选）。单类别抓斗任务中，令 G 为与任一 GT 的最大 RIoU≥0.5 的候选，B 为其余候选。g 是 G 内最高分类 logit，b 是 B 内最高分类 logit；t=log(0.05/0.95) 是现有输出阈值对应 logit。新增项为：

`L_select = mean_image relu(m + max(b, t) - g)`

B 为空时竞争值取 t；G 为空或无有效候选时，该图新增项取可反传的零并记录次数，不能强行将坏框标为好框。无 GT 图像本草案仅用原检测损失处理，并单独报告其数量；不能宣称因此改善了无目标误报。整 batch 平均含跳过图像的零项，避免有效样本数变化导致每图权重隐式放大。IoU、候选分组、max 索引均不通过框回传；梯度只通过选中分类 logits。并列采用固定 flatten 顺序。

设计初值建议固定 `m=0.1`（logit 单位）、`lambda=0.05`，总损失为原检测损失加 lambda×L_select。这是未验证的工程初值，不是依据 TEST 或已知最优值选择；首轮不扫描。阈值 0.05 和 RIoU 0.5 沿用既有定义。该项在合格候选已以足够 margin 胜出时为零；低于阈值时有提升 g 的梯度，错误候选领先时有提升 g、压低 b 的梯度。不保证训练收敛、保留旧输出或消除误报。

| 对比对象 | 实质差异/重叠 |
| --- | --- |
| SymNFL/SymPOLA | 保留原分类、几何权重和分配；新增项直接比较最终候选竞争条件。但选中的合格候选可能被 SymPOLA 标为负，存在梯度冲突，需在实现短检查中记录频次和相应梯度方向 |
| PQA pairwise | 历史实现对采样的 PQA quality 排序；本草案作用于原始 K1 分类 logit，并把固定输出阈值纳入同一个损失。二者都属于候选排序思想，不声称方法学首创 |
| DINO/S7 | 不使用 ROI 教师排序或教师任务输出；候选来自 K1 密集头 |
| V5 | 不使用对象—背景特征关系、分类适配器或 DINO 缓存；这是 GT 任务监督实验，不称为蒸馏 |

**A/B 对照**：两组都从同一 K1 epoch20 初始化，使用原生 K1 结构（不新增 adapter），real train+sim train，24 epoch、batch2、frozen_stages=1、SGD lr0.0025/momentum0.9/wd0.0001、clip10、warmup1000、step[16,22]，相同种子、数据变换及 checkpoint 保存方式。A 用原检测损失，B 仅多 L_select。新增损失通过正常分类路径更新 retina_cls、FPN 和未冻结主干；不直接更新 retina_reg，但共享特征更新仍可能改变几何，不能称为几何完全受保护。两组推理结构和阈值相同。

**实现时的一次必要短检查**：验证候选坐标/层级/logit 对应和 padding 规则；手构成功、低分、排序失败、无合格候选、空候选/空 GT 边界；确认新增项梯度路径和实际参数更新；在少量 source-train 上记录选中 g 的原分配标签及新增项与检测项的分类梯度关系、有效监督比例、运行时间和显存。这是链路与冲突检查，不是用几步训练判断长期有效性，不扩展成新一轮完整审计。若没有有效监督或存在实现错误，先停止正式训练；不能暗中改变分配规则来消除冲突。

**评价与停止规则**：source VAL 仍为738帧，按已有选权规则只扫描16/18/20/22/24，不为本方案另改选权公式。对选定 A/B 报告 success(RIoU≥0.5)、有输出帧、中心命中率（仅输出帧）、新增/丢失正确帧、RIoU、MCML和最长无输出区间，分 real/sim 汇总。覆盖与排序追踪复用已有工具的规则，不重新设计整套审计。最低观察门槛为 success 净增且输出覆盖、连续失败不恶化；这是工程筛选条件，不是单种子统计显著性保证。735/738 来自既有 V5 A，不作为新 A 的预定结果，source 容易且余量小应作为限制。首轮未改善则停止该设定，不顺势扫权重或加模块。TEST 仅在 source 选权和方案固定后作最终报告，不能用来决定 margin 或 loss weight。

当前交付止于设计草案；尚无实现验证或新训练结果。与历史独立质量头存在明确实现差异，但不足以保证研究价值或性能收益。

### K1 候选选择 A/B 实现与服务器操作（2026-09-27）

**当前状态：已封存，未开展正式训练。** 下列训练、VAL 与 TEST 命令保留为原设计记录，不是当前执行指令。最终决定及证据见本文末尾“候选选择 V1 封存结论”。

本地实现文件：`mmrotate/models/losses/candidate_selection.py` 提供单图选择项；`mmrotate/models/dense_heads/sym_eood_head.py` 在正式检测损失中接入，不改变 `simple_test`；`crane_project/configs/crane_symeood_k1_candidate_selection_{a,b}_v1.py` 为同预算 A/B；`crane_project/tools/preflight_k1_candidate_selection_v1.py` 完成配置、权重身份、少量 source 图的真实梯度检查。此段取代上节“未实现”的状态描述。它是 K1 的 GT 任务监督，不依赖 DINO 教师、cache 或 adapter。

代码合同：候选顺序按每层 H×W×anchor 展平并跨层拼接，复用 K1 的 padding 规则和 `bbox_coder.decode(..., max_shape=img_shape)`；解码与 RIoU 分组不对回归参数求导。每幅图的 loss 先计算，含无有效监督图像的零项一起按 batch 大小取均值，再乘固定 `0.05`。训练日志记录有合格候选、有梯度、合格候选被 SymPOLA 标为负、无合格候选和空 GT 的图像数。新项梯度到分类头、FPN 和未冻结主干；不直接进入回归卷积。预检针对固定的四张 source-train 图而非全量诊断，若均无新增项梯度，将输出 `NO_ACTIVE_SELECTION_SIGNAL_ON_SAMPLED_SOURCE`，不能将其视作正式训练收益证据。若有梯度则输出 `READY_FOR_FORMAL_TRAINING`，还需查看标签冲突、分类梯度余弦、显存。

本机仅有 PyTorch，无 MMCV/CUDA。选择损失及 source VAL 汇总共五项本地测试通过，相关 Python 文件编译及 `git diff --check` 通过；实际模型构建和 source 梯度由服务器预检负责。同步上述实现、配置与预检脚本后，在仓库根目录运行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
test ! -e work_dirs/crane_symeood_k1_candidate_selection_a_v1
test ! -e work_dirs/crane_symeood_k1_candidate_selection_b_v1
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/preflight_k1_candidate_selection_v1.py \
  --gpu 3 --out-json work_dirs/k1_candidate_selection_v1_preflight.json
```

预检会核对 K1 `epoch_20.pth` 的既有 SHA256、A/B 模型仅差选择项、24 epoch 与训练设置、无 DINO cache，并报告四张 source 样本的候选监督是否激活及梯度范围。逐图记录主头分类、主头回归和新增选择损失的实际量级及比值。标签统计分三层：任一合格候选被分为负、最高分合格候选被分为负、以及**新增项激活时**最高分合格候选被分为负。第三项才表示直接梯度冲突；`conflict_over_30_percent_warning` 用第三项除以激活帧数。它只基于四帧短样本，不是统计可靠的硬阈值，更不自动改候选筛选或权重。预检只读权重，不执行优化器更新。首轮 margin 仍固定为 `0.1` logit；提议的 `0.5–1.0` 会改变已预登记的单变量对照，不能在这次检查中直接替换。训练是否启动按下文补查结果判断。正式 A/B 均使用相同两卡；`tools/dist_train.sh` 已固定传入 `--seed 0`，两组均沿用它，不加 `--resume-from` 或 `--auto-resume`。

首次服务器预检的实际结果（2026-09-27）：四张 source-train 图均有合格候选，四张的选择损失和激活计数均为零，状态 `NO_ACTIVE_SELECTION_SIGNAL_ON_SAMPLED_SOURCE`。K1 checkpoint SHA256 与配置合同核对通过；但没有激活梯度可供这四张图验证，`active_conflict_fraction=null` 表示分母为零，不能解读为无冲突。先做**一次**有上限的 source-train 补查：固定种子 1701，在 real 和 sim 各抽 64 张，并排除首次四张；不读取 VAL/TEST，不更新模型、不改 margin 或 loss weight。补查只在每个域首次激活时验证分类头、FPN、回归卷积的梯度，逐图统计激活和损失量级。输出新 JSON，不覆盖原报告：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/preflight_k1_candidate_selection_v1.py \
  --gpu 3 --supplement-per-domain 64 \
  --out-json work_dirs/k1_candidate_selection_v1_source_supplement.json
```

若补查找到激活图像、梯度范围正确且标签冲突可接受，再执行以下原定 A/B 训练。若 128 张仍全为零，**停止当前 B 正式训练**：这只说明固定抽样下没有观察到新增监督，不证明 2781 张训练图全部为零，也不证明以后检测损失更新后永远不会激活。它已不足以支持把本次 24 epoch 当作已验证的“有效新监督”对照；记录为初始化监督过稀的限制，后续若重设计，应单独预登记新损失与新配置，不在这次实验中临时提高 margin、修改阈值或用 TEST 选样本。

首次四张和 128 张补查都是模型链路检查，并非训练效果证据。原始已满足条件的样本产生零选择损失是预期行为；真正需要核对的是在更多 source 样本上是否至少出现可作用的场景。

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_candidate_selection_a_v1.py 2 \
  --no-validate
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_candidate_selection_b_v1.py 2 \
  --no-validate
```

两组训练结束并确认 epoch 16/18/20/22/24 的 checkpoint 存在后，在物理 GPU 3 串行扫描 source VAL：

```bash
for arm in a b; do
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
    crane_project/tools/ckpt_sweep.py \
    --config crane_project/configs/crane_symeood_k1_candidate_selection_${arm}_v1.py \
    --work-dir work_dirs/crane_symeood_k1_candidate_selection_${arm}_v1 \
    --sweep-dir work_dirs/crane_symeood_k1_candidate_selection_${arm}_v1/source_val_sweep_protocol_v2 \
    --epochs 16 18 20 22 24 --gpu 3 || break
done
```

两组都使用原生 K1 推理路径；B 配置中的选择项只在 `loss()` 使用，不增加推理参数。按既有 source VAL 选权结果报告新增/丢失的正确帧、覆盖、仅有输出帧的中心命中、RIoU、MCML 和最长无输出区间。若需要对设计假设做定向核验，在两组 source VAL sweep 均完成后执行以下只读追踪；它会核对选中 checkpoint、PKL 的哈希与生成记录、738 帧序列及重放 top-1 与保存结果的一致性：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  crane_project/tools/trace_k1_candidate_selection_source_val_v1.py \
  --gpu 3 --out-json work_dirs/k1_candidate_selection_source_val_trace_v1.json
```

追踪报告按总体及 real/sim 分域列出 A 有输出/B 无输出、B 有输出/A 无输出、正确帧新增/丢失、stage 转移及三帧 V5 C 历史案例 `sim_seq10_00211/00212/00213` 在**新 A/B**中的实际 stage。它们不是新 A 的已知失败帧，也不能预设必须转成 success。分数项采用**每帧最高分合格候选和最高分错误候选**，分别给出有效帧数、均值，以及两组共同有效帧上的配对 B−A 变化；`bad_minus_good_score` 越低表示排序越有利于合格候选。全部是选权后的诊断项，不参与新的阈值或 checkpoint 选择；“A 零流失/B 新增输出”和“好框涨、坏框降”是希望观察到的模式，不能作为预设结果。

此阶段不调用 `--run-final-test`。只有 source VAL 选权和研究判断固定后，才分别用保存的选权文件做一次最终 TEST：

```bash
for arm in a b; do
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
    crane_project/tools/ckpt_sweep.py \
    --config crane_project/configs/crane_symeood_k1_candidate_selection_${arm}_v1.py \
    --work-dir work_dirs/crane_symeood_k1_candidate_selection_${arm}_v1 \
    --sweep-dir work_dirs/crane_symeood_k1_candidate_selection_${arm}_v1/source_val_sweep_protocol_v2 \
    --final-test-from work_dirs/crane_symeood_k1_candidate_selection_${arm}_v1/source_val_sweep_protocol_v2/sweep_results.json \
    --gpu 3 || break
done
```

`--final-test-from` 会核对 source VAL 选权文件、配置和选中 checkpoint 的身份；TEST 结果只作最终报告，不反向调整阈值、margin、loss weight 或选权。

### 候选选择 V1 封存结论（2026-09-27）

本轮为 **K1 原始分类分数的 GT 候选选择监督**，不是蒸馏：不使用 DINO 教师、教师缓存、特征对齐或分类适配器。目标是在存在 RIoU≥0.5 候选时，让最高分合格候选同时超过错误候选与固定输出阈值，并留出 margin。当前仅完成实现和无参数更新的 source-train 可用性检查，未完成正式 A/B 训练，不能记作新的蒸馏负结果或训练后性能结论。

证据：首次四张固定样本，加上种子1701的 real64张、sim64张补查（排除首次四张），共132张不同 source-train 图，约占2781张的4.7%。全部有合格候选，新增选择损失均为零；补查报告 `active_by_domain={real:0, sim:0}`、两域真实梯度检查次数均为0。对应设置为 K1 epoch20、margin=0.1、loss_weight=0.05 和当前输入变换。补查 checkpoint SHA256 为 `3ab0885159294beb820da1445c38045a342fd4956c3d094eeaaadf78deb745c2`，与预检一致。

标签统计已完成：132张均存在某些合格候选被分为负样本，但最高分合格候选被分为负的计数为0。没有新增项激活，因此没有验证激活时的真实梯度路径或梯度冲突；`active_conflict_fraction=null` 不能解释为冲突率为零。

**封存决定：按既定停止规则，暂停本版24 epoch A/B，不运行其后续VAL或TEST，不调整margin、权重或阈值。** 保留代码、配置和报告用于复核，不删除或覆盖。该决定基于未在已检查样本观察到新增监督，不证明全部2781张、其他变换或后续检测训练中始终不会激活，也不证明学生容量达到上限。

产物：`work_dirs/k1_candidate_selection_v1_preflight.json` 与 `work_dirs/k1_candidate_selection_v1_source_supplement.json`。后者已下载到本地 `/Users/mac/Downloads/k1_candidate_selection_v1_source_supplement.json` 并核对汇总与128条记录。

后续**不自动追加全量扫描**。只有明确需要回答“固定初始化与固定输入协议下，全体source-train的激活比例是多少”时，才单独开展一次只读全量统计；找到10张即停止不能得到全量比例，找到激活样本也不能证明训练收益。当前无此追加执行安排。新实验需要另行说明可验证且不重复历史方法的假设。对于本项 hinge 损失，减小margin只会使激活更少或不变；增大权重不能激活原本为零的项，不以这两种做法绕过封存决定。

### 教师—学生 source VAL 能力差距对照 V1（2026-09-27，已实现未在服务器执行）

候选选择 V1 封存后，研究主线回到“替换 DINO 时保留哪些教师能力”。新增只读工具 `crane_project/tools/audit_teacher_student_source_val_gap_v1.py`，比较同一738帧 source VAL 上的一个教师 frame-outcomes JSON 与 K1 学生 prediction PKL。它不运行推理、不读取 TEST、不改阈值、不选 checkpoint，也不把结果直接当作蒸馏收益。

工具先核对738个标注帧、教师 `seq/frame` 身份和学生一类 `max_per_img=1` 输出，再计算教师正确/学生错误、教师错误/学生正确、双方正确和双方错误四类，并按 real/sim 分域。学生框的 RIoU由GT和PKL重新计算；教师仅使用其报告中的 `top1_hit`、`top1_riou` 和 `top1_score`。因此它只能回答框级能力差距，不能证明教师背景响应、中间特征或学生候选阶段的差距。

教师角色必须在命令中显式写出。现有 `source_interpolation_result.json` 是 native-DINO 检测小头的 source 输出报告，不等同于 V1–V5 使用的冻结 DINOv2 特征缓存教师；若后续取得真正对应的教师任务输出，应使用相同工具但更换输入并保留角色标记。没有学生 provenance sidecar 时，工具按排序后的source VAL标注与PKL顺序配对并在报告中标记这一限制；有 sidecar 时会核对其 PKL SHA256。

运行后的决策只按以下顺序解释：若教师正确而学生错误的帧主要是学生已有合格候选但排序/分数失败，才补取最小的教师任务响应并设计分类分支蒸馏；若学生根本没有合格候选，才考虑空间特征迁移；若教师优势很少或教师与学生角色不匹配，则停止把该报告作为新蒸馏依据。四类计数不能单独授权训练，教师错误帧必须保留GT检测监督。

### 教师—学生差距报告复核与身份纠正（2026-09-28）

用户提供的 `teacher_student_source_val_gap_v1.json` 中，675双方正确、61学生独有正确、2教师独有正确、0双方错误均可逐帧复算。教师677/738，学生736/738；教师独有两帧为sim_seq10_00211/00212，学生均无输出。real学生226/226输出且中心正确，sim510/512输出且510帧中心正确；条件中心命中率均100%，输出覆盖应另报。

**撤回上次命令的epoch20身份归因。** 命令实际读取 `source_val_epoch24_results.pkl`（SHA256 `361ccd8848c162e92477e04e38f87cd7f28209f07129823e84045e8ec74031c9`，历史记录为epoch24），却把参数写成epoch20。旧代码只保存参数字符串，没有验证生成关系。这是命令和实现错误；该报告只能保留为历史产物描述，不能作为当前epoch20对照，也不能通过改文件名或改checkpoint字段补造身份。

教师来自native-S14检测头alpha=0.5的source报告。其frame_outcomes的top1_hit是几何命中，deployment_top1_hit另加部署分数门槛。该alpha的原始汇总中二者恰好都是677，但部署有1帧静默，不能把每个有top1_score的条目都当部署输出。原文件没有教师框，不能重算教师中心命中、坐标/变换和GT绑定。native-DINO任务教师与旧特征缓存教师不同，这本身不是任务响应蒸馏的错误；后续要明确拟替换的是哪一个完整检测系统。

原工具只实现框级结果对照，未完成计划中的K1候选阶段分层。当前两帧空输出不能直接归因为score_threshold；V5 A/C候选追踪也不能替代epoch20的候选证据。

修复后输出协议为 `teacher_student_source_val_gap_audit_v2`（保留原脚本文件名）：拒绝epoch24 PKL与epoch20 checkpoint的明显冲突；检查提供checkpoint的文件和SHA；有生成sidecar时核对预测/配置/权重/标注SHA及source_val split；没有sidecar时默认拒绝，只有显式 `--allow-unverified-student` 才输出历史描述报告。增加教师已选alpha、source协议、命中与RIoU一致性、数值范围、real_seq07=226/sim_seq10=512及完整帧集合检查；空类别也输出零。即使身份sidecar存在，教师输出口径和输入变换证据仍不完整，因此报告不自动批准蒸馏。五项针对性回归测试及本地738帧历史复算通过，未连接服务器、未重跑模型。

下一步只补目标K1 epoch20的预测生成绑定：优先用已有source VAL选权记录、epoch20 PKL和sidecar；没有可绑定的产物才考虑一次固定epoch20的source推理。历史教师输出继续复用。取消先前“少于5帧就停止蒸馏”的任意门槛；双方都正确的帧也可能含有可迁移信息。当前61/2仅不支持无差别模仿教师最终框，不能证明DINO能力不可迁移。无需为了重新得到同一61/2计数而在服务器重跑修订脚本。

### 教师—学生差距 V2 实际结果（2026-09-28）

epoch20 学生的 source VAL 预测已经通过 sidecar 绑定：PKL SHA256 为 `9bfa8195dcbf42218d182107177c0db9f8210797b354b2cf6d8c33679072de57`，checkpoint SHA256 为 `3ab0885159294beb820da1445c38045a342fd4956c3d094eeaaadf78deb745c2`，预测配置为 `crane_symeood_k1_source_val_eval.py`，738条运行顺序身份和 source 标注哈希均已核对。教师仍是 native-DINO alpha=0.5 source top-1 报告，不是DINOv2缓存特征教师。

逐帧结果为：双方正确674，教师正确/学生错误3，教师错误/学生正确61，双方错误0。教师几何命中677/738，epoch20学生几何命中735/738。三帧教师独有正确为 `real_seq07|117`、`sim_seq10|211`、`sim_seq10|212`，三帧学生均为空输出；学生错误恰好只有这三帧。教师独有样本数量很少，且教师在61帧上错误而学生正确，说明这个 native-DINO 最终框输出不能直接作为学生监督目标。报告中的 `suitable_for_epoch20_distillation_decision=false` 正确保留了这一证据边界。

V2修复了V1的epoch身份错误，但仍是框级最终输出比较。它没有学生候选阶段、教师中间特征或教师分类响应，不能判断三帧空输出发生在候选生成、分数阈值还是排序阶段。下一步不直接蒸馏这3个最终框，也不读取TEST挑选困难样本；若继续替换DINO，应使用与V1–V5相同的DINOv2特征教师，先在部署学生的对应分类/候选位置取得教师响应，再进行短链路监督检查。

### 旧版服务器测试入口修复（2026-09-28）

服务器环境是 MMCV/MMDetection 1.x；`crane_project/tools/test.py` 原先却是 MMEngine 2.x 入口，导入 `mmdet.utils.register_all_modules` 时失败。该入口现改为委托仓库根目录的旧版 `tools/test.py`，保留原命令路径并使用项目已有的 provenance 选项。source VAL 推理必须使用 `crane_symeood_k1_source_val_eval.py`，不能用普通 K1 配置的 TEST split。

### provenance 兼容修复（2026-09-28）

服务器 `tools/test.py --result-identity-out` 生成的 sidecar 协议为 `mmdet_runtime_result_order_identity_v1`，它用 `runtime_dataset_order`、`result_count`、配置和 checkpoint 哈希绑定结果，不包含 `split` 或顶层 `annotations_sha256`。教师—学生差距工具此前只接受 ckpt_sweep 的另一种 provenance 结构，误报 `Student provenance source split/annotations mismatch`。现已同时支持两种 sidecar：旧版 runtime identity 校验738个 frame_key顺序、PKL/配置/权重哈希；ckpt_sweep provenance 继续校验 source_val 和标注哈希。该修复不放宽身份检查。

### 教师—学生差距审计的配置身份纠正（2026-09-28）

epoch20 source VAL PKL 的 runtime sidecar 绑定的是 `crane_symeood_k1_source_val_eval.py`，因为该配置把 `data.test` 指向738帧 source VAL；`crane_symeood_k1.py` 的 `data.test` 指向固定 TEST。审计命令若传入普通 K1 配置，会得到 `Student provenance config mismatch`，这是应保留的身份保护。工具现输出 sidecar/config两侧SHA和实际配置路径，便于区分配置传错与权重传错。后续命令必须传入 `crane_symeood_k1_source_val_eval.py`。


## 合并的历史记录

以下记录按原文保留实验细节与证据入口。其“当前状态”“下一步”等表述只反映各记录写成时的状态；后续裁决以本文较新的日期记录为准。此次合并没有重跑实验或重新核验外部产物。


### 语义蒸馏 V1 设计快照

来源：原 `docs/dino_to_symeood_semantic_distillation_v1.md`。

## DINO 到 SymEOOD 的几何保护语义蒸馏 V1

### 目的

本入口验证一个有限问题：冻结 DINOv2 已表现出的语义候选能力，能否在不保留
DINO 推理链的前提下，改善普通 SymEOOD K1 的分类排序。它不是 Base V3
时序 refiner 的替代实验，也不使用固定 TEST 选择权重或超参数。

### 实现边界

- 教师输入来自既有 DINOv2 ViT-L/14 离线特征缓存；训练过程不运行 DINO。
- 只读取 source TRAIN（`train:train` 与 `train_sim:train`），缓存必须通过文件身份、
  模型名、通道数、有限值和完整性预检。
- 教师特征固定为 FP16 `64x64`，只蒸馏 FPN 第 0 层对应的分类塔特征，最多
  使用 4096 个空间 token。非正方形 DINO patch grid 先按右侧/底部规则补成
  正方形，再缩放到 `64x64`，避免直接拉伸破坏与学生画布的空间对应。
- 蒸馏区域由训练 GT 的 OBB 外接矩形给出；教师张量始终 detach。
- 原版 head 没有多层分类塔，因此增加零初始化的 `1x1` 分类残差适配器；蒸馏
  支路在适配器输入前 detach。蒸馏梯度只能更新分类适配器与训练期投影器，不能
  更新 backbone、FPN 或回归分支。原有 SymKLD、角度定义和 OBB 目标不变。
- 推理前导出学生 checkpoint，移除 `semantic_distillation.*`，保留轻量分类
  适配器。导出后的权重由 `crane_symeood_k1_dino_semantic_student_v1.py`
  加载，推理不实例化 DINO 或蒸馏投影器。

### 公平比较

1. GT-only：普通 `crane_symeood_k1.py`，相同 source 数据、seed 和训练轮次。
2. 蒸馏 V1：只增加前景分类塔语义蒸馏。
3. 两者分别在 source VAL 选择 checkpoint；固定 TEST 只做一次冻结评估。

必须同时报告完整 OBB、中心、尺度、角度、小目标切片、推理时间和显存。语义
排序改善不能替代尺度与角度结果。若 source VAL 无改善或几何指标退化，停止该
路线，不根据 TEST 调整 `loss_weight`。

训练批量、优化器、基础学习率、warmup、epoch 数和学习率衰减点继承普通 K1：
`samples_per_gpu=2`、SGD `lr=0.0025`、24 epochs、warmup 1000 iterations、
epoch 16/22 衰减。蒸馏组与 GT-only 组的唯一预期模型差异是分类适配器及其训练损失。

### 当前状态

代码入口和本地单元测试已完成；尚未在服务器完成缓存完整性检查、训练、学生导出
和评估，因此尚不能声称性能提升或形成论文贡献。

缓存预检的峰值进程内存只打印到终端，不进入稳定 JSON。重复检查的证据相同
时保留首份报告及其 SHA256；缓存身份或完整性发生变化时仍拒绝覆盖。


### 语义学生与 K1 同口径比较

来源：原 `docs/dino_semantic_student_fair_comparison_20260923.md`。

## DINO 语义蒸馏学生与普通 K1 的同口径比较

### 固定设计

- 两组训练均使用原有的 source train / train_sim 数据和 24 epoch 日程；蒸馏学生只在训练时读取冻结 DINO 特征，评估使用学生配置 `crane_symeood_k1_dino_semantic_student_v1.py`。
- 两组均只从 `epoch_16/18/20/22/24.pth` 选择权重。source VAL 共 738 帧；固定 TEST 不用于选权。
- `ckpt_sweep.py` 沿用历史两阶段规则：跨域加权中心召回率距离候选最优值不超过 0.005，最长连续漏检不超过 5 帧；合格候选按 `0.35*TDR + 0.25*R_center + 0.20*sim/ACI + 0.20*(1-min(sim/A-RMSE,90)/90)` 排序。跨域权重为 sim 0.7、real 0.3。没有合格候选时沿用旧脚本的两级 fallback，并在结果中显式记录。
- `eval_crane_offline.py` 的 `mode='test'` 用于输出完整时序指标；在 source VAL 阶段，数据来源仍然是 `val/annfiles`，不是固定 TEST。
- 两组均通过相同的 `tools/test.py`、DOTA 转换和当前 `eval_crane_offline.py` 指标版本计算。选择完成后才分别对各自选中权重运行一次固定 TEST。

### 历史结果与修订结果的关系

旧 `work_dirs/crane_symeood_k1/ckpt_sweep/sweep_results.json` 保持原样。它来自指标修复前的离线评估器；不能与当前评估器生成的蒸馏结果直接比较。使用该文件保存的普通 K1 五组 VAL 预测在本地以当前评估器重算时，`epoch_24` 的 `sim/A-RMSE(deg)` 从旧报告的 1.6529 变为 9.0415，`real/mean_RIoU` 从 0.9026 变为 0.769。相同五组预测在当前指标版本和原选权公式下，初步重选为 `epoch_20`。这说明旧 `epoch_24` 是历史选择，不应被悄悄改名为当前指标版本下的选择。

#### K1 固定 TEST 指标下降的已确认原因

本地保存的普通 K1 `epoch_24` 固定 TEST 预测为 992 帧，PKL 与 DOTA 文件逐帧数量匹配，导出前后中心最大差约 0.0066 像素。real 域 420 帧中只有 274 帧有输出，另 146 帧缺测；274 帧中有 272 帧满足 15 像素中心阈值。旧报告的 `real/R_center=99.27%` 是 `272/274`，当前 `64.76%` 是 `272/420`。这不是同一个分母，也不是新模型突然失去 34.51 个百分点的已输出帧定位能力。

旧 `real/mean_RIoU=0.8046` 是有输出帧上的轴对齐近似值。当前旋转框 IoU 在有输出帧上为 `0.7042`，把 146 帧缺测记为零后是 `0.4594`。仿真域 572/572 帧均有输出，旧近似 IoU `0.9087`，当前真实旋转 IoU `0.8761`。旧近似计算不能作为修订版 RIoU；当前全帧指标也不能被误读为有输出帧的定位质量。

上述拆解来自只读审计 `crane_project/tools/audit_k1_metric_denominators_v1.py`，本地明确分母和序列覆盖率的报告保存在 `work_dirs/crane_symeood_k1/metric_compatibility_audit_v1/epoch_24_r4.{json,md}`。审计还核对了 PKL 与 DOTA 导出的几何一致性，最小 IoU 为 0.99817，排除了导出时大幅改变框几何的解释。服务器新选中的 K1 `epoch_20` 审计显示 real `420` 帧、输出 `277`、缺测 `143`、输出后中心命中 `276/277=99.64%`、全帧中心检测召回 `276/420=65.71%`；已输出框的严格旋转 IoU 为 `0.7129`，全帧零填充值为 `0.4702`。缺测帧没有可计算的中心误差。

#### 缺测阶段和进一步诊断

普通 K1 的实际主头推理不运行 NMS：先用 `score_thr=0.05` 过滤每个特征层候选，再从剩余候选取最高分的一个。因此在原配置下最终为空，表示没有候选通过这个分类分数门槛；不能把空输出归因于 NMS。现有最终 PKL 不保存低于门槛的候选，不能仅凭它断言放宽门槛后会得到正确框，更不能据固定 TEST 调整门槛。

本地历史 K1 `epoch_24` 的 real 缺测按序列为 `seq02:107/220`、`seq03:39/200`；较长连续段包括 `seq02:129–172`（44 帧）、`seq02:2–41`（40 帧）。这是缺测集中出现的证据，不是暗光或小目标的因果证明。已有 source VAL `epoch_20` 预测仅缺 `real 1/226` 和 `sim 2/512`，所以仅在 source VAL 上放开固定分数门槛的诊断样本很少，不能用它直接决定新的线上门槛。新增的 `audit_k1_source_val_threshold_v1.py` 只比较原 `0.05` 与诊断用 `0` 的配对预测、检查已有框完全一致、统计补出的框是否真正命中；它不选门槛，不读取固定 TEST，也不授权改动线上策略。

服务器必须在独立 `ckpt_sweep_metric_v2` 目录生成两组新的选权和 TEST 报告；保留旧结果。普通 K1 的 VAL 预测可从历史 `ckpt_sweep` 缓存重算指标，蒸馏学生则需运行五个候选权重的 VAL 推理。两组报告都要核对 checkpoint、VAL 标注、预测和指标协议身份。若普通 K1 缺少候选 checkpoint 或缓存预测，应停止比较并补齐来源，不用旧指标填补。

### 解释边界

同口径 TEST 对照必须比较当前重新选择的普通 K1 与蒸馏学生。历史冻结 `epoch_24` 的既有结果仍可作为历史参考，但它使用旧指标版本选权，不能与新选权协议混成同一组公平对照。VAL 和 TEST 的数据角色应分别标注。只有 TEST 计算完成后，才能判断蒸馏是否改善检测表现；训练损失下降和训练钩子的 `save_best` 不能代替该结论。

### 固定 TEST 配对审计入口（2026-09-23）

已有同口径 TEST 结果显示蒸馏学生在 real 域的全帧中心命中率低于普通 K1。两份终端输出分别记录 849 和 794 个总预测框，但终端汇总尚不能说明哪些帧被学生补回、哪些帧由 K1 独有，也不能用不同输出集合上的条件平均 RIoU 直接判断几何改善。

只读入口 `crane_project.tools.audit_k1_dino_student_paired_test_v1` 复用已有两组 `final_test_metrics_v2.json`、`results.pkl` 和 `Task1_grab` 文件，不重新推理。它先核对 TEST 标注、报告与预测哈希以及 PKL 到 DOTA 的逐帧几何一致性，然后按 real/sim 与序列输出三种中心命中分母、全帧 RIoU、双方独有正确帧和共同输出帧的配对 RIoU。完整 JSON 保留 992 帧逐帧记录；Markdown 只呈现汇总。

该审计用于解释已暴露 TEST 的差异，不能据逐帧得失选择新的训练样本、阈值、checkpoint 或在线切换规则。下一版蒸馏应先在 source 数据上预设保留 K1 原有能力的设计与对照，再按既定选权协议检验。当前本地缺少学生 TEST 原始预测，因此不能把从终端总框数推算的逐域计数标为已完成的配对审计。


### K1 warmstart V2 方案快照

来源：原 `docs/dino_k1_warmstart_distillation_v2_20260923.md`。

## K1 初始化的轻量蒸馏配对实验 V2

### 已有证据与本轮目的

已暴露的固定 TEST 配对审计 `k1_dino_student_paired_fixed_test_audit_v1` 表明：从头训练的语义特征蒸馏学生在 real 域从 K1 的 277 帧输出降至 222 帧；`real_seq03` 失去 60 帧 K1 原有输出，`real_seq02[2,41]` 与 `[137,169]` 两段均未恢复。共同输出帧在 `seq02` 有一定 RIoU 收益，但不能抵消缺测代价。审计 JSON 的本地 SHA256 为 `5ef70baf2897f1a79fc62cfbaa4b13ede95b6afab12435d484174d6be785f6f3`。这些 TEST 结果仅解释失败，不参与 V2 的训练样本、阈值和 checkpoint 选择。

V2 只回答一个问题：**从 source VAL 已选中的普通 K1 初始化后，原有的 DINO 前景特征损失是否相对同预算普通微调产生增益？** V2 不是新的 DINO 检测头候选蒸馏，也没有解决教师候选的 source 支持不足问题。学生推理仍只运行 SymEOOD；训练时从磁盘读取已有冻结 DINO 缓存，不运行 DINO 大主干。

### 两组唯一的计划差异

| 项目 | 无 DINO 损失对照 | DINO 特征蒸馏 |
|---|---|---|
| 配置 | `crane_symeood_k1_dino_warmstart_control_v2.py` | `crane_symeood_k1_dino_warmstart_distill_v2.py` |
| 初始化 | source VAL 选中的 `crane_symeood_k1/epoch_20.pth` | 同一文件 |
| 学生结构 | K1 + 零初始化分类残差适配器 | 完全相同 |
| source 数据 | train + train_sim | 完全相同 |
| DINO 缓存数据流 | 读取，用于匹配数据和运行条件 | 读取，供蒸馏损失使用 |
| 训练 | 4 epoch；SGD，学习率 0.00025；每卡 batch 2；2 GPU 顺序运行 | 完全相同 |
| 唯一有意差异 | 无 DINO 损失 | 原 V1 前景特征蒸馏损失，权重 0.05 |

先运行 CPU preflight：要求 K1 checkpoint SHA256 与新版 source VAL 选权结果、已生成的 K1 TEST 身份报告一致；检查 train 与 train_sim 两份缓存收据；解析两组配置；构建两组 CPU 模型并加载同一 K1 权重，确认新增分类适配器为零、初始分类和回归输出完全相同。若任何检查失败，停止训练，不回退到旧版选权结果。

4 epoch、低学习率是预先固定的微调预算。两组 checkpoint 只在 source VAL 以同一 `ckpt_sweep.py` 规则选择；固定 TEST 每组只运行一次。对照的含义是区分 DINO 损失与普通继续训练的影响；与原 24 epoch 从头训练 K1 的比较则仍须单独标明训练预算不同。不得依据已暴露 TEST 的特定帧挑选样本或调节损失。

### 服务器顺序

1. 同步本文件、三个 V2 config、preflight 工具和专项测试。保留原 V1 权重与报告。
2. 运行 `tests/test_k1_dino_warmstart_v2.py` 和 `tests/test_semantic_feature_distill_v1.py`。
3. CPU preflight 成功并输出 `MATCHED_WARMSTART_READY` 后，先训练 control，再训练 distill。仅使用 GPU 0、1，不并发训练。
4. 用统一学生推理配置 `crane_symeood_k1_dino_semantic_student_v1.py` 对两组 epoch 1–4 做 source VAL 选权；核对两个 `sweep_results.json` 的协议、候选和选择。
5. 各自按选权结果运行一次固定 TEST，再生成配对审计。正式评价同时报告全帧中心命中、输出覆盖率、共同输出帧几何、序列连续缺测，以及推理时间/显存；不把 TEST 当作优化输入。

本地没有服务器 K1 epoch 20 权重或 MMCV/MMRotate 运行环境，因此 CPU 模型构建与真实权重加载必须在服务器 preflight 中完成。本地验证只覆盖契约逻辑、配置文本、语法和已有蒸馏模块测试。


本轮 warmstart V2 的归档结果、配置身份纠正与后续迁移方向见 本文「2026-09-23 蒸馏交接与正式结果」。


### 2026-09-23 蒸馏交接与正式结果

来源：原 `docs/dino_distillation_handoff_20260923.md`。

## DINO → SymEOOD 初步蒸馏试验与对话迁移记录（2026-09-23）

### 当前结论与任务定位

当前阶段是轻量学生蒸馏可行性验证。目标是在训练时利用冻结 DINOv2 教师，部署时只运行 SymEOOD 学生。先前运行时调用 DINO 的 scoped 融合、DINO 独立检测诊断，与本轮缓存特征蒸馏是不同实验身份。

本轮 warmstart V2 相对于同预算无蒸馏对照，只增加 1 帧中心命中，real 最长连续缺测均为 62 帧；未获得实质连续检测收益。结论仅适用于这套前景特征损失、初始化和训练预算，不等于所有蒸馏方法无效，也没有多随机种子统计显著性结论。

### 试验目的、设计与针对困难

历史关注 real_seq02 远距片段 [2,41]、暗光片段 [137,169] 与 real_seq03 小目标片段 [129,192]。这些是已暴露 TEST 的诊断标签，不允许据此挑训练样本或调整参数。seq03 抓料可能影响形状，但不从正式 TEST 删除，也不把动作影响当作已证实因果。

从头训练特征蒸馏 V1 曾丢失 K1 输出。V2 回答：从已选中 K1 初始化，DINO 特征损失是否比普通继续训练有额外收益？两组均从 K1 epoch_20 初始化，计划训练4 epoch，SGD lr=0.00025、momentum=0.9、weight_decay=0.0001，warmup100步、ratio0.1、step=[3]，梯度裁剪10。source train2033帧 + train_sim748帧；VAL738帧；TEST992帧。计划每卡batch2、两卡顺序训练；这些训练设置来自仓库配置，包内没有训练日志，未据此验证实际启动参数。

两组学生均有零初始化分类残差适配器；control关闭蒸馏损失。distill读取1024通道DINOv2 ViT-L/14缓存，使用单层FPN（feature_level=0）、前景余弦对齐、权重0.05、最多4096 tokens。protect_geometry=True 对蒸馏输入执行detach，蒸馏梯度直接更新适配器和训练期投影，不直接更新主干/FPN；检测损失仍更新检测网络，因此不能声称几何完全受保护。尚未蒸馏抓斗教师RPN/ROI的任务输出。

源域VAL从epoch1–4按既有自定义规则选权，两组均选epoch_1。原规则包含加权中心召回约束、MCML≤5及TDR、中心、ACI、角度软评分；本次未依据TEST重选。当前metric_v3只是输出目录名，报告metric_protocol_version仍为2。

### 正式 TEST 表（直接读取归档JSON）

R_center为全部GT帧中的中心命中比例，阈值15px；mean_RIoU为缺测计零的全帧值。条件定位误差与覆盖率必须同时解释。MCML是当前评估协议下连续失败统计，不是直接训练目标，也不能自动继承教师数值。

| 指标 | control | distill |
|---|---:|---:|
| real/R_center(%) | 72.38 | 72.62 |
| real/mean_RIoU | 0.5074 | 0.5085 |
| real/DFR(%/frame) | 2.9759 | 2.9855 |
| real/ACI | 0.9253 | 0.9252 |
| real/TDR_w10(%) | 79.13 | 79.13 |
| real/MCML_max(frames) | 62 | 62 |
| real/MCML_mean(frames) | 25.33 | 25.33 |
| real/MCML_pass(limit=5) | 0 | 0 |
| real/MRF(frames) | 7.53 | 7.53 |
| sim/A-RMSE(deg) | 4.8098 | 4.7998 |
| sim/R_center(%) | 100.0 | 100.0 |
| sim/mean_RIoU | 0.8284 | 0.8282 |
| sim/DFR(%/frame) | 2.2792 | 2.2759 |
| sim/ACI | 0.9562 | 0.9562 |
| sim/TDR_w10(%) | 100.0 | 100.0 |
| sim/MCML_max(frames) | 0 | 0 |
| sim/MCML_mean(frames) | 0.0 | 0.0 |
| sim/MCML_pass(limit=5) | 1 | 1 |

### 配对结果（完整记录见 paired_test_audit.json）

| 分组 | 对照输出 | 学生输出 | 对照中心命中 | 学生中心命中 | 共同输出RIoU：对照→学生 |
|---|---:|---:|---:|---:|---|
| real | 309 | 310 | 304 | 305 | 0.689634719 → 0.689516215 |
| real/seq02 | 121 | 121 | 117 | 117 | 0.685973641 → 0.685302251 |
| real/seq03 | 188 | 189 | 187 | 188 | 0.691991052 → 0.692228394 |
| sim | 572 | 572 | 572 | 572 | 0.828363051 → 0.828201647 |

学生独有输出记录：`[{"frame_key": "real_seq03_00186", "domain": "real", "sequence": "seq03", "frame": 186, "gt_short_edge_px": 22.40755271911621, "baseline": {"output": false, "center_hit": false, "riou_hit": false, "center_error_px": null, "riou": null}, "student": {"output": true, "center_hit": true, "riou_hit": false, "center_error_px": 6.655543107343824, "riou": 0.49407209503787036}}]`。

real共有309帧双方输出，学生共同帧RIoU略降；seq02双方均缺99帧，输出集合完全相同；seq03仅补回1帧。real的RIoU命中总数均为274，不能将新增中心命中直接称为新增完整框命中。sim均完整输出，几何差异微小。

### 必须纠正的历史说明

1. 旧选权SHA 551a255db55e0125141d35239e207298868326e5909895470dcefb527bf2f4ac 与本地统一学生推理配置 crane_symeood_k1_dino_semantic_student_v1.py 完全一致。此前把它解释为“选权后配置被修改”，证据不足；更符合现有证据的是VAL使用学生推理配置，而提供的TEST命令错误换成训练配置。原设计文档也要求用统一学生配置选权。哈希保护本身正常，但之前给出的命令和原因解释有误，不应为消除保护而重写旧哈希。

2. 用户粘贴过real R_center=96.19%、RIoU=0.7937、MCML=6的历史表。本包未定位到这一整组报告的可靠身份，之前直接确认其为独立native-S14 DINO正式TEST是不成立的。包中formal_integrated_nms05与formal_integrated_test_refactor绑定scoped配置及BrightAug epoch20，包含融合结果：real条件中心97.39%、另列全帧25px中心72.38%、MCML39。它们不能当作同口径独立教师上限。需要对应96.19/6的原始报告、配置、权重身份及预测，才能重算同协议教师比较。

3. 实际配对文件名是 work_dirs/crane_symeood_k1_dino_warmstart_v2_paired_test_audit.json（及.md），此前漏写crane_symeood前缀导致打包失败。

### 当前改进方向与停止条件

本轮停止继续围绕这一个TEST净增帧调节余弦损失、阈值或训练轮数。保留当前负结果及同预算对照。下一步候选方向是任务相关的选择性蒸馏，但尚未证明有效，也未在本轮实现。

先确定实际教师身份和源域教师正确、学生不足的监督支持，复用已有source质量与覆盖诊断；不要重复把源域缺少真实小目标验证支持包装成新发现，不重新划分数据。存在可信监督时，以K1初始化，保留GT检测损失，优先迁移对象与邻近背景的区分或可靠类别响应；异构检测头需明确区域/候选对应，不直接复制权重或全面照搬教师框。

可考虑源域清晰教师视图与学生合理光照扰动的对应，以及有限后段学生参数接受蒸馏梯度。加入原有能力保留约束是待验证选择，不保证覆盖率保留；全部新增机制不能一次堆叠。教师几何监督需经source GT核验，小目标分辨率/尺度问题独立讨论。新方案先固定设计、同预算对照、source VAL选权，再冻结评价；已暴露TEST只能诊断，不能声称未见测试的无偏确认。

资源目标：训练优先缓存教师输出，不加载在线大教师计算图；控制缓存与GPU张量上界，顺序运行实验。最终单学生推理必须测延迟和峰值显存，不能凭参数少宣称加速；本包没有新延迟测量。

### 证据位置、验证与复现

本地仓库 /Users/mac/Documents/paper/symEOOD；服务器 /media/omnisky/personal_files/ljj/symEOOD；Python3.8 mmrotljj。原始压缩包 /Users/mac/Downloads/k1_dino_warmstart_test_analysis_20260923.tar.gz。SHA256为47a9c479d8f6e1ed8ff81013e945071f833ac315d56f9a99f4ae27c57dfdabe9。全部包内成员路径与哈希保存在相邻evidence/dino_warmstart_v2_20260923/archive_manifest.json。

关键原文已复制到 [证据目录](../evidence/dino_warmstart_v2_20260923)，不会依赖/tmp解压目录。control_selection.json、distill_selection.json保存选权；control_test.json、distill_test.json保存正式指标；paired_test_audit.json保存992帧；两份formal摘要只作为历史身份核对证据。

本轮实际验证：压缩包SHA与服务器记录一致；两组选权checkpoint哈希与TEST及配对审计一致；config哈希三方一致；配对审计绑定的TEST文件SHA与实际文件一致；配对记录992帧。未重跑训练、模型推理或GPU测试；缺少PKL/Task1_grab与权重，不能独立重算框几何或验证实际训练参数。

复现入口：ckpt_sweep.py --final-test-from 必须使用选权时同一config并保持config哈希，sweep目录必须与JSON所在目录一致。当前两组位于 work_dirs/crane_symeood_k1_dino_warmstart_{control,distill}_v2/ckpt_sweep_metric_v3/final_test/epoch_1/；配对工具 audit_k1_dino_student_paired_test_v1.py 使用 --gt-dir、两组 --*-report/--*-pkl/--*-pred-dir 读取已存在预测即可，不需要重跑模型。

### 新对话接续提示

先读取本文与证据JSON。用户当前目标是DINO能力迁移和学生轻量部署，暂不写论文正文。不要混淆DINOv2特征教师、抓斗DINO检测器、scoped运行时融合、Base V3观测系统和本轮warmstart学生。V2已经完成初步对照且没有实质收益；后续讨论任务相关蒸馏时先检查是否与历史失败方法重复。优先核实教师身份与监督支持，避免反复让用户重新选权、打包或重复完整诊断。


### 2026-09-27 课题迁移快照

来源：原 `docs/DINO课题迁移记忆_20260927.md`。

## DINO 优化与 SymEOOD 蒸馏：Claude 课题迁移记忆

整理日期：2026-09-27。当前研究任务由用户明确为 **DINO 上的优化**，包括 DINO 检测能力改进及向轻量 SymEOOD 学生迁移；本次接续不自动扩展到深度估计、可靠性策略或论文正文。

本文是便于上传 Claude 的**有日期的迁移快照**，正文包含独立理解课题所需的背景、结果和限制。原始实验仍在既有主线文档与结果文件中维护，本文不另立平行实验总账。历史材料中的“蒸馏尚未开始”已经过时：截至本快照，蒸馏已推进到 **V5 对象—邻近背景关系蒸馏，完成训练、source VAL 选权、固定 TEST 和产物审计**；完成实验不等于已经取得稳定性能收益。

### 1. 可直接复制到 Claude 记忆导入框的摘要

```text
[2026-09-27] 我的课题是港口门座式起重机抓斗顶梁的单目、单类旋转框（OBB）检测。检测框围绕顶梁参考平面，不是整个抓斗或平台。当前工作集中于 DINO 优化及 DINO→SymEOOD 的轻量学生蒸馏。
[2026-09-27] DINO 所针对的三个困难是：暗光大目标的分类分数/排序崩塌；远距离目标的低置信度输出不足；更小旋转目标的空间采样、候选覆盖与最终质量排序问题。代表诊断片段分别是 real_seq02[137,169]、real_seq02[2,41]、real_seq03[129,192]。这些 TEST 片段已暴露，不能用于选权、调阈值或选训练样本。
[2026-09-27] 冻结 DINOv2 ViT-L/14 加 rotated RPN/ROI 小头已提供暗光语义救援证据：BrightAug 的 0/33 到 ScopedDINO 的 29/33，加因果尺度/角度稳定后 32/33。scope 使用过 target 信息，这些是机制诊断；不是通用学生成绩或未知视频泛化。
[2026-09-27] 历史正式 DINO 组件是 native S14、ROI 分类权重插值 alpha=0.5、S7 disabled、ROI NMS=0.5。S7 高分辨率分支把小目标诊断片段候选 R@100 从 55/64 补到 64/64，但该方案最终 Top-1 仍为 50/64。候选覆盖、正确选框、框几何和连续性必须分别评价。
[2026-09-27] 已尝试分类权重插值、S7 readout/RPN、NMS 调整、pairwise/relative/highres/unified 排序、时序 ROI projector、native spatial adapter 和 K1/DINO 几何融合。部分 source 指标改善，但有旧正确帧损失或未迁移到固定困难片段；不能简单说全部无效，也不能拼接不同模型的最好数字。
[2026-09-27] 蒸馏学生使用 SymEOOD K1 加零初始化分类残差适配器，教师为冻结 DINOv2 的离线特征。推理仅运行学生，不调用 DINO。蒸馏 V1 丢失原 K1 输出；V2 K1 warmstart 相对同预算对照仅增加 1 帧中心命中；V3 放开 FPN 蒸馏梯度、V4 修正前景掩码后，均未获得输出覆盖或中心命中的额外收益。
[2026-09-27] 最新 V5 已完成 24 epoch 对象—邻近背景关系蒸馏 A/C 对照。A 无关系损失，C 增加权重 0.05 的关系 MSE；均由同一个 K1 epoch_20 初始化，source VAL 分别选中 A epoch_24、C epoch_18。V5 的 frozen_stages=1，不是 V3/V4 的整个 ResNet-50 冻结。
[2026-09-27] V5 固定 TEST 的 real 420 帧：A/C 输出 255/258 帧，中心命中均为 253 帧；条件中心命中率为 99.22%/98.06%，全帧中心召回均为 60.24%。C 新增与丢失中心命中各 22 帧，seq02 净减 15 帧、seq03 净增 15 帧。最长连续无输出为 39/64 帧；RIoU 连续失败 MCML 则均为 64 帧。sim RIoU 从 0.8871 降到 0.8724。当前结论是实验完成，但尚未证明稳定的整体收益。
[2026-09-27] R_center 的分母要明确：中心命中/有输出帧是条件定位率；中心命中/全部 GT 帧是全帧召回。输出覆盖率要单列。历史 MCML 表示 RIoU<0.5 的连续失败，不等于连续无输出。缺测计零的 mean_RIoU 与有输出帧条件 RIoU 也不能混用。
[2026-09-27] source train 为 real 2033 + sim 748=2781 帧；source VAL 738 帧；固定 TEST 992 帧（real 420、sim 572）已暴露。独立真实困难验证不足是已知限制，不要反复要求重做同一全量诊断，也不能把它说成唯一已证实原因。新方案应先说明与既有失败方法的差异，并在 source 上确定设计和同预算对照。
[2026-09-27] 我希望保留已有工作、负结果和证据边界；不要编造结果、来源或贡献。低风险本地操作按授权持续完成，数据删除、对外发送等遵守授权范围。重要结论注明模型、数据、日期和来源；需要核对的地方明确标记。
```

这段记忆是本次对话和项目证据的整理，不代表已经读取或导出了用户在其他 ChatGPT 会话中的全部个人记忆。

### 2. 三个问题究竟是什么

| SymEOOD 面临的问题 | DINO 介入的理由与已完成工作 | 目前的证据边界 |
| --- | --- | --- |
| **暗光大目标：有几何候选，但分类分数和排序失效。** 暗光下目标不一定消失，正确候选可能被背景高分候选压制，产生长段漏检。代表片段 `real_seq02[137,169]`，33 帧。 | 引入冻结 DINOv2 的语义特征和 source 训练的小检测头；ScopedDINO 将历史 BrightAug 的 Top-1 `0/33` 提高到 `29/33`，因果框稳定后达 `32/33`。 | 该 scope 使用过 target-dev 信息，只支持暗光救援机制；稳定器改善宽高和角度，不补造输出。V5 蒸馏学生在同一 33 帧仍无输出，不能说救援能力已迁入学生。 |
| **远距离目标：低置信度和输出覆盖不足。** 普通 K1 曾在 `real_seq02[2,41]` 连续无输出，共 40 帧。 | DINO 逐级候选审计推翻了“始终没有可用候选”的笼统解释。历史 highres 方案在该片段 R@100 为 `40/40`、Top-1 为 `38/40`；另一时序方案曾为 `39/40`。 | 这些是不同 DINO 实验的诊断结果。远距已不再是该 DINO 候选生成支线的独立瓶颈，却仍可能是 SymEOOD 学生的覆盖问题；V5 A/C 在这 40 帧分别输出 `5/0` 帧。 |
| **更小旋转目标：细节不足，补出候选后仍选不准。** `real_seq03[129,192]` 共 64 帧，历史短边约 `1.124` 个 DINO token。 | S7 高分辨率 readout/RPN 与 ROI 读出把 R@100 `55/64→64/64`，说明可以补足候选；随后尝试多类质量排序和时序方案。 | 同一 highres 方案 Top-1 仍为 `50/64`，剩余问题是最终 OBB 质量排序。抓料/形变可能影响观测，但不能把未经验证的解释当因果结论，也不能从正式 TEST 删除失败帧。 |

三个困难的共同目标是增加**正确且连续的检测输出**，同时保留已有几何和正常帧能力。DINO 本身的计算成本是另一个工程约束，促使研究转向训练用教师、推理用学生。[来源 S1、S2、S5]

普通 K1 的现有主头推理先按 `score_thr=0.05` 过滤候选，再取最高分一个框，**不运行 NMS**。因此不能把 K1 空输出解释为 NMS 抑制；DINO RPN/ROI 的 NMS 诊断属于另一条检测链。最终 PKL 不含低于阈值的候选，也不足以证明降低阈值就会恢复正确框。[S2]

### 3. 已做过的 DINO 改进及保留结论

| 路线 | 改动与正向结果 | 代价、状态与后续应记住的内容 |
| --- | --- | --- |
| 冻结语义救援 + 因果框稳定 | 冻结 DINOv2 ViT-L/14，训练 oriented RPN/rotated ROI 小头；等价 OBB 表示对齐后，对 log 宽高、双角周期方向做因果 EMA，历史暗光 Top-1 `0→29→32/33`。 | EMA 系数 `0.25`；中心、分数、排序和是否输出不变。target-derived scope 使结果只具诊断资格。 |
| ROI 分类器权重插值 | `alpha=0.5` 时 source Top-1 `662→677/738`，新增 15、丢失 0。 | 这是分类权重插值系数，不能与 EMA 系数混同；更大 alpha 会丢旧正确帧。作为历史正式 native-S14 组件保留。 |
| S7 readout/RPN 与高分辨率 ROI | 补足极小目标候选；source-safe highres 版本达 `688/738`、small `311/350`、lost=0；固定 small R@100 达 `64/64`。 | 候选增益没有变成该片段 Top-1 增益；正式 native-S14 基线仍不启用 S7。 |
| DINO NMS `0.1→0.5` | 同一 137 帧追踪中，NMS 后有可用候选的帧数 `117→128`。 | Top-1 仍 `117/137`；新增保留的 11 帧转为最终排序失败。不能继续把阈值扫描当主要解法。 |
| Pairwise、连续/相对质量、highres/unified ranker | relative-quality 达 source `691/738`、lost=0；unified hard-pair 达 `696/738`、lost=1。 | 相对质量/highres 在固定 small 仍未得到所需提升；unified 只通过 bounded-risk research gate，不能称 exact-safe，也不补写缺失的 target 结果。 |
| 时序 ROI projector/候选接管 | 对象跨帧检索 margin 改善。 | 闭环检测曾降到 `425/738`、lost=263；身份相似不等于 OBB 几何质量。不能把余弦相似度直接当定位质量。 |
| native spatial adapter V3 | source 4 epoch 可训练，轻量 head 资源可控。 | epoch1 `+1/-2`、epoch2–4 `+2/-4`，回退 epoch0；与 relative-quality 的最小互补审计只多 1 帧 oracle，停止该选择器路线。此 V3 不是下节蒸馏 V3。 |
| K1/DINO 几何融合及 seq11 replay | K1 锚框、DINO fallback、bounded residual、retention/history 等尝试；retentive V3 epoch9 在 source 有改善。 | TEST 未稳定复现；seq11 V4 replay epoch10 是历史候选，困难块 OOF 未完成。不能把这条融合支线当作已完成的轻量蒸馏。 |

历史 BrightAug QFL、RegQuality/PQA 等失败路线也已封存。后续方案需要说明新增信息或监督，不能仅更名重跑。以上数字来自 S1 的对应实验记录，本次没有重新运行这些模型。

### 4. 蒸馏已完成到哪一步

#### 4.1 教师、学生和目的

教师是**冻结 DINOv2 ViT-L/14 特征**，通过离线缓存提供训练监督；不是把抓斗 DINO RPN/ROI 的最终框、类别分数或时序输出整套复制给学生。学生是 SymEOOD K1（ResNet-50/FPN/旋转检测头）加零初始化分类残差适配器。当前蒸馏的意图是把语义能力迁到学生分类表示，保留原 GT 检测损失及 OBB 任务。

V1–V4 使用前景特征余弦对齐和训练期投影器；V5 改为对象与邻近背景的关系图匹配，不再启用旧特征损失。学生推理配置保留分类适配器，不实例化教师或训练期损失。**学生推理评估链已运行；独立部署包验收、端到端延迟和峰值显存改善尚无本次证据。** 不依据“移除了大教师”直接宣称已达到实时速度。[S2–S5]

#### 4.2 V1–V5 演进与结果

| 版本 | 实验目的和主要改动 | 已完成结果与裁决 |
| --- | --- | --- |
| **V1：从头训练特征蒸馏** | K1 加分类残差适配器；前景余弦对齐；辅助梯度在 FPN 输入处 detach，仅更新适配器与投影器。 | 历史配对记录：real 输出从 K1 的 `277` 降到学生 `222`；seq03 丢失 60 帧 K1 原输出，远距和暗光两段未恢复。完成了初步实验，未保住原有能力。原 V1 设计文档的“尚未训练”是旧状态。[S2、S3] |
| **V2：K1 warmstart 同预算对照** | 两组同由 K1 epoch20 初始化；4 epoch，lr=0.00025；control 不加蒸馏，distill 只增加原前景损失，权重0.05。 | 均选 epoch1。real 输出 `309→310`，中心命中 `304→305`；全帧中心召回 `72.38%→72.62%`；RIoU 命中均274；MCML 均62。仅 seq03 第186帧新增中心命中，其 RIoU≈0.494，未新增完整框命中。无实质整体收益。[S4] |
| **V3：FPN 梯度范围 A/B/C** | 三组冻结整个 ResNet-50（frozen_stages=4）；A 无蒸馏，B 阻断蒸馏进入 FPN，C 放开；均 K1 epoch20、4 epoch。 | 均选 epoch1；同一877帧有输出（real305、sim572），中心命中集合相同；real 全帧中心召回70.71%、mean_RIoU0.5003、MCML38相同。损失、参数和浮点预测确有差异，但特征对齐改善未转为任务收益。旧 PKL 缺生成时 provenance，保留追溯限制。[S5、E3] |
| **V4：修正掩码的 A/C 对照** | 修正旋转框前景外接范围计算，缩小掩码时以最大池化保留小目标；沿用 V3 4 epoch 预算，A 无损失、C 蒸馏进入 FPN。 | source 探针证明修复和梯度路径生效；正式 A/C 均选 epoch1，real 输出均305/420，中心命中均297，条件中心命中率97.38%；MCML均38。掩码修复后仍无额外覆盖/中心收益。工程修复不追溯改变 V3 权重。[S5、E4] |
| **V5：对象—邻近背景关系蒸馏** | 由 K1 epoch20 warmstart，恢复24 epoch日程；A/C同预算；C增加对象/背景关系MSE，权重0.05。 | A选epoch24、C选epoch18，完成992帧固定TEST和2026-09-27产物审计。出现序列间收益/退化交换，未形成稳定整体收益，详见第5节。[S5、E1] |

每版内部对照用于判断该版蒸馏的额外收益。V2、V3/V4、V5 的冻结范围与训练预算不同，不能根据各版 TEST 的最高值再选择“最终最佳版本”。

#### 4.3 V5 的具体实现及曾经的短探针

V5 在变换后的 GT 旋转框内部构造对象掩码；邻近背景来自外扩区域，留一格间隔、外扩至四格，并排除 padding。教师和学生各自对对象特征求原型，再计算原型与各位置特征的余弦关系。对象区域、背景区域的关系 MSE 等权，整体乘 `0.05`。它迁移的是对象—背景关系，**不是教师候选框质量、类别 logits 或时序轨迹**。[代码 C1]

配置中 `protect_geometry=True` 使该辅助关系项只经分类适配器更新，阻断它直接进入基础 FPN；原检测损失仍训练可训练主干阶段、FPN 与检测头，所以不能声称整个训练过程保持回归几何不变。V5 为 `frozen_stages=1`，与 V3/V4 的 `frozen_stages=4` 不同。[C1、E1]

两组均读取同一教师缓存以匹配数据管线；A 不使用关系项，C 使用。共同日程为24 epoch、SGD lr=0.0025、momentum=0.9、weight decay=0.0001、warmup1000、step=[16,22]、梯度裁剪10、每卡batch2。source VAL候选为epoch16/18/20/22/24。[E1、C2]

在正式训练之前，V5短source探针已经运行，但结果是 `no_support_in_this_probe`，**不是已通过蒸馏收益门槛**：教师区分度存在（real AUC=1.0，sim约0.984），但学生相对control的对象—背景分类gap略降；real关系MSE相对改善约0.119%，sim略差。短探针与正式24 epoch训练是不同实验；正式结果仍按正式A/C对照解释。[E2]

### 5. 最新完成的 V5：核对后的结果

本节直接读取并复核2026-09-27完整审计JSON的992条逐帧记录。A是无关系损失对照，C是关系蒸馏学生；中心阈值15 px，RIoU命中阈值0.5。以下结果属于**已暴露固定TEST的描述性诊断**。

#### 5.1 real 总体和 sim 几何

| 指标 | A：epoch24 | C：epoch18 | 解释 |
| --- | ---: | ---: | --- |
| real输出帧数 / GT帧数 | 255/420 | 258/420 | 净增3帧输出 |
| real输出覆盖率 | 60.71% | 61.43% | 不等于正确检测率 |
| real中心命中帧数 | 253 | 253 | 总数没有提高 |
| real条件中心命中率：命中/有输出 | 99.22% | 98.06% | C下降约1.15个百分点 |
| real全帧中心召回：命中/全部GT | 60.24% | 60.24% | 历史报告的 R_center 即此口径 |
| real RIoU≥0.5命中帧数 | 239 | 238 | C少1帧 |
| real全帧mean_RIoU，缺测记零 | 0.4433 | 0.4460 | 小幅增加，不代表每个已输出框更准确 |
| real有输出帧条件mean_RIoU | 0.7301 | 0.7261 | 输出集合不同；不能直接归因于同帧几何变差 |
| real双方共同输出236帧的mean_RIoU | 0.7391 | 0.7442 | 同帧几何有小幅改善 |
| real最长连续无输出 | 39帧 | 64帧 | C变差；按序列与帧号连续性重算 |
| real MCML：最长连续RIoU失败 | 64帧 | 64帧 | 两组最差失败段所处序列不同 |
| real TDR_w10 | 67.43% | 69.47% | 部分时序统计改善 |
| real MCML_mean | 34.67帧 | 33.67帧 | 历史同协议指标 |
| real MRF | 16.45帧 | 10.53帧 | 历史同协议指标 |
| real DFR（%/frame） | 2.4967 | 2.5817 | C更高 |
| real ACI | 0.9430 | 0.9355 | C更低 |
| sim输出/中心命中 | 572/572 | 572/572 | 两组均完整输出与中心命中 |
| sim mean_RIoU | 0.8871 | 0.8724 | C几何退化 |
| sim角度RMSE | 1.3104° | 1.7880° | C角度误差增加 |

#### 5.2 总数相同不代表逐帧相同

C相对A新增22个中心命中，同时失去22个A原有中心命中。共同输出236帧，A独有输出19帧、C独有22帧。中心命中得失统计包含“仍有框但定位由对变错/由错变对”，因此不必等于独有输出数量。

| 序列 | 输出帧数 A→C | 中心命中 A→C | RIoU命中 A→C | 结果 |
| --- | --- | --- | --- | --- |
| real_seq02，220帧 | 122→110 | 122→107 | 122→107 | 中心命中净减15；最长无输出延长 |
| real_seq03，200帧 | 133→148 | 131→146 | 117→131 | 中心命中净增15；局部改善未覆盖seq02代价 |

A最长无输出为 `real_seq02[133,171]` 的39帧；C为 `real_seq02[110,173]` 的64帧。A的64帧连续RIoU失败位于 `real_seq03[129,192]`，C的64帧连续RIoU失败位于 `real_seq02[110,173]`。**旧MCML最大值相同，掩盖了失败位置和无输出长度的变化。**

#### 5.3 回到原来的三个困难片段

以下由同一V5审计的逐帧记录计算，仅用于解释，不据此继续调参。

| 固定诊断片段 | A：输出/中心命中/RIoU命中 | C：输出/中心命中/RIoU命中 |
| --- | --- | --- |
| 远距 seq02[2,41]，40帧 | 5 / 5 / 5 | 0 / 0 / 0 |
| 暗光 seq02[137,169]，33帧 | 0 / 0 / 0 | 0 / 0 / 0 |
| 更小目标 seq03[129,192]，64帧 | 15 / 13 / 0 | 24 / 23 / 9 |

V5的确在这个小目标片段产生了局部正确框，但暗光没有恢复，远距退化。可保留“局部能力变化”的观察；不能称为三个问题已解决、DINO能力完整迁入学生或整体有效改进。

#### 5.4 审计已经支持什么，还有什么没有证明

完整审计记录两组各5个source VAL候选的checkpoint/PKL/provenance核对，以及固定TEST和逐帧几何对应检查。记录中的checkpoint历史config可用，报告为无配置差异。A选中epoch24、C选中epoch18，与source选权一致。

C日志有312条关系损失记录，均为正，范围0.00068–0.00158，最后0.00084；A无关系损失记录。两组分类适配器均已非零。但正损失和适配器改变不能单独证明该辅助项对最终改善的因果贡献。

A目录存在两份均到epoch24的日志；checkpoint与唯一训练日志的运行绑定仍未完全确认，不能把两份日志相加称为48 epoch。审计也没有历史完整梯度轨迹。本次整理复算JSON内计数、比例及失败区间，并核对证据副本哈希；没有重新访问服务器权重、加载PKL、训练或运行模型推理。

### 6. Claude 接续时必须保留的口径与状态

1. **模型身份分开。** DINOv2特征教师、native-S14抓斗DINO检测组件、ScopedDINO、K1/DINO融合refiner、蒸馏学生均为不同模型。native spatial adapter V3、retentive geometry V3、FPN蒸馏V3不是同一个V3；关系蒸馏V5也不是OBB可靠性V5.1。
2. **K1权重有历史版本。** 当前蒸馏初始化是重新按修订指标source VAL选出的普通K1 epoch20；旧K1 epoch24及BrightAug epoch20不能替代它。epoch20 SHA为 `3ab0885159294beb820da1445c38045a342fd4956c3d094eeaaadf78deb745c2`。
3. **数据角色固定。** source train real2033+sim748=2781，source VAL738，固定TEST992（real420+sim572）。TEST已暴露；不据其单帧得失挑样本、阈值、epoch或损失参数，不把反复诊断称为未知测试确认。
4. **指标分母显式写出。** 条件中心命中率、全帧中心召回、输出覆盖率分别报告。RIoU条件平均与缺测计零平均分别报告。MCML与最长无输出分别报告，跨序列和帧号断点重置。
5. **区分实际执行和代码存在。** 已有V1–V5实验与学生推理评估，不再误写“蒸馏未启动”；但蒸馏成功、统一实时部署、多随机种子稳定性和未知真实视频泛化仍无充分证据。导出脚本存在不等于部署包已验收。
6. **已有支持限制不用反复重新发现。** 历史尺度定义下source-train有真实小目标，source-val小目标350帧均为sim、real为0；“native错而S7对”监督稀疏属于指定候选协议。不能说训练集完全没有真实小目标，也不能将验证不足当作唯一已证实原因。
7. **测量有效性不等于模型优化。** seq03[130,187]的58帧抓料/未离料区间可作预定measurement-valid敏感性分析，但完整TEST保留；不能只剔除失败帧来证明收益。
8. **当前阶段定位。** 本次用户要求接续DINO优化。后续需在已有V1–V5负结果和局部收益上说明新机制，采用匹配对照及source验证；不自动回到seq11融合、可靠性策略或深度支线，不自动启动新训练。

建议 Claude 在读完后先复述：三个困难是什么、哪一层已经改善、哪一层仍失败、V5完成了什么，以及三个最重要的证据限制。把第1节摘要用于记忆，把全文放入课题项目资料，后续按新日期更新事实。

### 7. 来源和复核入口

| 编号 | 文件 | 本文用途 |
| --- | --- | --- |
| S1 | [冻结DINOv2与SymEOOD检测融合](冻结DINOv2与SymEOOD检测融合.md) | 三个困难、历史改进、模型身份、2026-09-22整体复盘；蒸馏状态须结合后续记录 |
| S2 | 本文「语义学生与 K1 同口径比较」 | K1输出/指标分母纠正、V1比较与选权协议 |
| S3 | 本文「K1 warmstart V2 方案快照」 | V1失败记录、V2同预算设计 |
| S4 | 本文「2026-09-23 蒸馏交接与正式结果」及[归档证据](../evidence/dino_warmstart_v2_20260923) | V2正式指标与配对结果；其中“下一步关系蒸馏尚未实现”已被V5完成状态更新 |
| S5 | 本文 V3–V5 实验记录 | 覆盖V3、V4、V5，历史快照更新至2026-09-27 |
| E1 | [V5完整产物审计](../evidence/dino_claude_handoff_20260927/v5_artifact_audit_complete.json) | 992帧记录、A/C权重身份、条件指标、失败区间与限制 |
| E2 | [V5短source探针](../evidence/dino_claude_handoff_20260927/v5_source_probe_r2.json) | `no_support_in_this_probe` 的实际结果 |
| E3 | [V3产物审计](../evidence/dino_claude_handoff_20260927/v3_artifact_audit.json)、[A](../evidence/dino_claude_handoff_20260927/v3_a_test.json)、[B](../evidence/dino_claude_handoff_20260927/v3_b_test.json)、[C](../evidence/dino_claude_handoff_20260927/v3_c_test.json) | V3三组同输出集合及指标 |
| E4 | [V4 A报告](../evidence/dino_claude_handoff_20260927/v4_a_test.json)、[V4 C报告](../evidence/dino_claude_handoff_20260927/v4_c_test.json) | 掩码修复后正式TEST指标 |
| C1 | [关系损失实现](../../mmrotate/models/losses/object_background_relation.py)、[检测器接入](../../mmrotate/models/detectors/sym_eood_detector.py) | 关系定义、掩码、辅助梯度范围 |
| C2 | [V5共同配置](../../crane_project/configs/crane_symeood_k1_dino_object_background_relation_common_v5.py)、[C配置](../../crane_project/configs/crane_symeood_k1_dino_object_background_relation_c_v5.py)、[学生推理配置](../../crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py) | 训练日程、关系项、推理身份 |

原始证据按字节复制到相邻目录，没有改写报告结论。来源路径、压缩包成员和SHA256见 [source_manifest.json](../evidence/dino_claude_handoff_20260927/source_manifest.json)。V5完整审计SHA256为 `7bd637524c96f37b08dbe3aa3d0ed7dc5505f173aba1f5ee706785a0209a31e6`；protocol为 `k1_dino_object_background_relation_v5_artifact_audit_v3`。

本地仓库：`/Users/mac/Documents/paper/symEOOD`。服务器仓库：`/media/omnisky/personal_files/ljj/symEOOD`，历史运行环境Python3.8 / `mmrotljj`。这些路径用于复现定位，不表示Claude网页端能直接读取本机文件。

上传Claude时，单独上传本文即可获得完整文字背景；需要核查数值时再附E1及相关证据。本文引用的源码和文件路径是来源索引，历史命令不构成自动执行指令。
