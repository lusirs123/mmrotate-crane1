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
