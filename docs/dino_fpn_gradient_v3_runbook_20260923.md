# K1 → DINO 特征蒸馏的 FPN 梯度范围对照 V3

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

预检会核对 K1 `epoch_20.pth` 的既有 SHA256、A/B 模型仅差选择项、24 epoch 与训练设置、无 DINO cache，并报告四张 source 样本的候选监督是否激活及梯度范围。逐图记录主头分类、主头回归和新增选择损失的实际量级及比值。标签统计分三层：任一合格候选被分为负、最高分合格候选被分为负、以及**新增项激活时**最高分合格候选被分为负。第三项才表示直接梯度冲突；`conflict_over_30_percent_warning` 用第三项除以激活帧数。它只基于四帧短样本，不是统计可靠的硬阈值，更不自动改候选筛选或权重。预检只读权重，不执行优化器更新。首轮 margin 仍固定为 `0.1` logit；提议的 `0.5–1.0` 会改变已预登记的单变量对照，不能在这次检查中直接替换。确认报告 `status=READY_FOR_FORMAL_TRAINING`，并审阅冲突、梯度余弦与显存后，使用相同两卡、相同随机种子依次训练两组。`tools/dist_train.sh` 已固定传入 `--seed 0`，两组均沿用它；不要加 `--resume-from` 或 `--auto-resume`：

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
