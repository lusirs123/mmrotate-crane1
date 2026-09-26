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
  --out-json work_dirs/crane_symeood_k1_dino_object_background_relation_v5_artifact_audit_v2.json
```

核查报告中的 `c_minus_a_metrics` 用于记录 TEST 差值，`conclusion` 固定为本轮“real 时序有弱变化、中心命中率和 `MCML_max` 无改善”的证据边界。若核查失败，应先修复产物身份或日志问题，不重新解释模型效果。
