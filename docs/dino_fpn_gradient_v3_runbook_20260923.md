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
