# SymEOOD 几何精度优化交接（2026-10-01）

> 用途：在新对话中继续几何精度优化。本文汇总本轮对话、现有代码及收到的实验结果；保留事实、推断和待验证内容的区别。后续优先更新本文，不再为每次改动创建交接文件。
>
> **当前结论：保留 SymEOOD＋尺度增强 B。D 和固定E-H v1均未形成联合收益。E-H在VAL选epoch22，改善real覆盖/连续性及sim角度，但尺寸/RIoU退化；固定TEST上real全帧中心正确数与B相同、输出多1帧，real RIoU基本持平，sim角度及RIoU退化，不替换B。后续先补有限TRAIN/VAL机制证据，不据已多次暴露的TEST调参或重选权重，不恢复DINO、候选排序或完整审计，不立即重训。**

当前数据、已完成实验与论文素材已集中整理为[港口新数据与尺度增强实验总记录](detection/港口新数据集与EOOD_SymEOOD尺度增强实验总记录.md)。该主记录维护成果与论文口径，本文保留执行交接及后续设计；E仍只有设计，未实施。

2026-10-01文档整理补充：主记录第3节按实际代码说明SymNFL/SymKLD/SymPOLA及EOOD主头损失接入，第7节并入本文第15节的E机制、文献边界与验证门槛。本文公式中的`global_positive_count`指当前GPU batch跨FPN层汇总的正样本归一化（逐图至少计1），不是跨GPU all-reduce。此次只整理文档，未修改训练代码或重新运行实验；EOOD配置声明L1但本地主头未返回该项的发现，不能未经服务器历史版本核验就外推到已有训练。

## 1. 新窗口先读这部分

### 1.1 用户目标

用户希望完成小论文：以 EOOD 为原方法主基线，说明 SymEOOD 在真实港口检测覆盖与连续性方面的价值，并改善当前平均定位精度、仿真域角度精度及尺寸精度的代价；之后再评估预测框输入下的测量/深度可靠性。

早期目标是轻量化大 DINO 并保留其能力，尝试过 DINO→SymEOOD 蒸馏和融合。但用户目前明确：**暂不考虑后续 DINO，先把 SymEOOD＋尺度增强的几何精度做好。** 不为了论文复杂度强行加入 DINO。

用户接受已有方法的小范围适配，但希望有清楚的问题、合理机制和对照证据。不能把普通尺度增强、增加两个回归项或新的命名直接包装成原创算法，也不能保证论文发表。

### 1.2 已经回答的问题

- 新数据上长段漏检是真实模型现象；固定权重推理、PKL/TXT及候选追踪已核验过，不应重新归因于文件同步或导出遗漏。
- 恢复 seq06 后 SymEOOD 的 real 表现明显改善，但增加数据同时增加优化步数，不能将全部收益单独归因于目标尺度或 seq06 的某种外观。
- 等比例尺度缩小 B 有明显检测收益；光照增强 C 当前配置效果差，不与 B 合并使用。
- EOOD＋相同尺度增强也明显受益，因此普通增强不是 SymEOOD 独有创新；SymEOOD 没有全面领先。
- TRAIN/VAL 几何分解显示：SymEOOD B 的中心、尺寸精度存在问题，sim 的纯角度误差并非最明显的差距。
- 中心—尺寸补偿 D 已完成正式训练、VAL选权和TEST。VAL改善未稳定转化到TEST，D当前不替换B。

### 1.3 新窗口的工作边界

1. 先继承本文和现有结果，避免重复完整诊断、历史机制核查或固定权重链路核验。
2. 若需要补证据，只补决定下一项优化所必需的信息。不要自动要求新划分、全量审计或重跑DINO。
3. 新方案用TRAIN/VAL设计和选权，不根据TEST调整损失、阈值或选择其他epoch。
4. **当前TEST已多次暴露和诊断，不能宣称整个研究过程中它始终是完全未接触的独立测试集。**
5. 只在本地修改代码；用户自行在服务器运行。不要连接服务器。
6. 本文交接本身不授权下一版训练或代码改动。先解释设计，再按新对话中的授权推进。

## 2. 项目位置、命名与用户偏好

| 项目 | 路径/设置 |
|---|---|
| 本地项目 | `/Users/mac/Documents/paper/symEOOD` |
| 服务器项目 | `/media/omnisky/personal_files/ljj/symEOOD` |
| 服务器环境 | `mmrotljj`，Python 3.8，旧版 MMCV/MMDetection/MMRotate 栈 |
| 当前数据目录 | `crane_project/data/crane_grab_port_day2night_v1` |
| 已有数据/实验记录 | `crane_project/data/crane_grab_port_day2night_v1/README.md` |
| 旧DINO研究记录 | `docs/dino/DINO到SymEOOD蒸馏实验总记录.md` |
| 本地CPU验证解释器 | `/opt/anaconda3/envs/mmrot/bin/python` |
| 常用训练卡 | `CUDA_VISIBLE_DEVICES=2,3`，两张 GTX 1080 |
| 常用验证卡 | `--gpu 3`，物理第四张卡 |

用户偏好：持续完成已授权实现及必要检查，避免反复确认；代码修改后审查实际梯度、配置继承及运行入口；给可复制的服务器命令。不要一轮轮修审计代码后要求重跑。文档尽量集中，保留历史配置与结果身份，不能覆盖原始数据或修改历史SHA来绕过检查。

当前正式配置名虽然**不带 seq06 后缀，实际都包含恢复的 seq06**。曾经存在无06的同名历史实验，不能只按目录名判断身份，也不能合并两套目录。配置清理和改名快照在数据目录的 `provenance` 下。

## 3. 固定数据集及几何约束

### 3.1 当前主划分

| 集合 | 公开真实序列 | 自采港口序列 | 仿真序列 | real | sim | 总数 |
|---|---|---|---|---:|---:|---:|
| TRAIN | real_seq01：339 | real_seq05：560、real_seq06：466、real_seq12：141、real_seq13：304 | sim_seq08：748 | 1810 | 748 | 2558 |
| VAL | real_seq07：226 | real_seq14：149 | sim_seq10：512 | 375 | 512 | 887 |
| TEST | real_seq03：200 | real_seq04：668，夜间 | sim_seq09：572 | 868 | 572 | 1440 |

公开 seq02 不参与当前主划分，原数据仍保留。sim数据未改动。不要因当前TEST指标高低再移动序列。

旧版划分：TRAIN2781（real2033＋sim748），VAL738，TEST992（real_seq02＋real_seq03＋sim_seq09）。旧真实TRAIN为01/04/05/06。新划分将04移到TEST，加入自采12/13，恢复06；不是完全恢复旧训练集。

### 3.2 新增标注与采样

- seq12：`1 (13).mp4` 与 `1 (1).mp4` 连续内容合并；seq13：`1 (7).mp4`；seq14：`1 (10).mp4`。
- 新增序列3 FPS采样，X-AnyLabeling轴线标注，按 **k=2.1** 转OBB：长边为轴线长度、短边为长边/2.1。标注中央参考结构，不是整个张开的抓斗外轮廓。
- 此k不是模型K=1，也不是深度标定系数；不能称所有旧real标注都采用该比例。
- seq12图像1280×720；seq13/14为1920×1088。源视频有噪声、压缩和解码异常；seq14保留了已标注模糊帧，没有按指标挑除困难帧。
- 新编号不等于原视频绝对帧号。seq12的00121→00122约有9秒缺口，时序计算不能跨该点假设连续采样。
- 同港口昼夜场景可用于光照迁移评价，但不代表未知港口泛化；按原视频/连续采集事件分组，同场景本身既不能直接认定泄漏，也不能证明不存在近重复。
- 当前TEST seq04曾是旧训练数据，所以当前正式模型使用ImageNet初始化，不能加载接触过seq04的旧项目权重冒充无泄漏初始化。

### 3.3 深度估计约束

必须保持比例缩放及坐标可还原：训练中的图像、OBB中心/宽高使用一致变换；测试坐标还原到原图，深度计算使用原图对应内参。禁止未同步还原的裁剪、平移或不等比例拉伸。

可逆的是坐标映射，不是缩小后丢失的像素细节。保持比例也不代表预测框尺寸误差或深度误差已经得到保证。真实数据没有独立深度/摆角真值，检测框评价与端到端测量评价应分开；检测sim_seq08/09/10也不应直接等同于另一套几何标定/测量真值数据。

## 4. 训练和选权契约

当前增强与D对照沿用SymEOOD K1训练设置：

| 设置 | 固定值 |
|---|---|
| 初始化 | ImageNet ResNet-50，`torchvision://resnet50`；`load_from=None`、`resume_from=None` |
| 主干 | `frozen_stages=1`，不是整个backbone冻结 |
| 批量 | 双卡，每卡2张，总batch4；workers每卡2 |
| 训练 | 24epoch，seed0 |
| 优化器 | SGD，lr0.0025，momentum0.9，weight_decay0.0001 |
| 梯度裁剪 | max_norm10 |
| 学习率 | linear warmup1000，step `[16,22]` |
| 模型 | SymNFL、SymKLD、SymPOLA及原辅助分支保持原设置；SymKLD权重2 |
| VAL扫描 | 只扫描epoch16、18、20、22、24 |
| 硬约束 | 加权R_center≥本次最大值−0.005；MCML≤5 |
| 软评分 | 0.35×TDR＋0.25×Rc＋0.20×ACI＋0.20×AngleScore |
| 跨域加权 | sim0.7＋real0.3 |
| TEST | 固定VAL所选权重，通过 `--final-test-from` 执行 |

B、C、D正式训练都从原初始化开始，**不是从已经训练好的B或旧K1续训**。D短检查用B权重仅用于梯度链路检查，不改变正式初始化。

EOOD保留原模型后处理，可能输出多候选；SymEOOD固定score_thr0.05、max_per_img1。诊断按实际执行路径处理，不能给SymEOOD虚构RPN或NMS阶段，也不能为公平比较偷偷改EOOD后处理。EOOD离线静态统计沿用最高分第一行框，不按GT选框。

## 5. 指标口径：不能混用

1. **用户要求的正文中心命中率：仅统计有输出帧。** 同时另报输出覆盖率和全帧中心正确覆盖。
2. 现有 `ckpt_sweep.py` / `eval_crane_offline.py` 终端 `R_center` 仍采用**全帧分母，无输出计零**。当前固定阈值为15px；不要把98.85%直接改名为输出帧准确率。
3. 新几何对比脚本分别报告两种分母。中心距离、尺寸、纯角度误差基于输出框，RIoU汇总另有全帧口径。
4. 当前MCML是连续 **RIoU<0.5** 的失败长度，不等于连续完全无输出；需要时分别报最长无输出和最长RIoU失败。
5. 原协议角度RMSE对无输出或中心误差≥10px的帧计90°。纯角度RMSE没有这类罚项，必须明确区分。
6. 旧数据集中心分母、IoU及部分时序实现与当前不同，而且真实TEST序列不同。旧结果只作历史说明，不能与新结果直接计算提升比例。
7. 共同输出帧只用于公平几何比较；不能用它掩盖新增漏检。要一起报告完整集正确覆盖和输出增减。
8. 原图像序列采样间隔与缺口影响时序解释，当前按帧指标不自动等于真实时间上的控制可靠性。

## 6. 从旧数据到当前增强的结果脉络

### 6.1 新数据及seq06恢复

未加06的SymEOOD曾在seq03的100–199帧全部无输出。固定权重103帧重推理、200帧PKL/TXT核对及候选追踪一致，排除了这次导出链路遗漏。100帧中15帧有RIoU≥0.5候选但分数不足，85帧解码候选几何质量也不足；100帧全图最高分均低于0.05。不能简单只降低阈值。

恢复06后SymEOOD的seq03剩29帧失败，曾集中在164–192。用户指出抓斗在这些图中人眼可见，不能直接归因于“趴着抓料”或标注错误。后续新权重的同段检查发现分类与几何响应问题，推动尺度增强。旧K1 seq03单独MCML曾为12、输出161/200；不能仅凭旧结果断言新训练代码错误或新旧数据等价。

当前含06、无增强主对照（现有README记录）：

| TEST指标 | EOOD ep24 | SymEOOD ep20 |
|---|---:|---:|
| real 全帧中心正确覆盖% | 84.68 | 92.05 |
| real mean_RIoU | 0.7296 | 0.7636 |
| real TDR% | 89.18 | 96.24 |
| real MCML_max | 101 | 29 |
| sim mean_RIoU | 0.9296 | 0.8564 |
| sim 角度RMSE° | 1.5017 | 2.1024 |

恢复06的收益不能单独证明“图片小、目标像素大”是原因，也不能只归因于样本数。当前不要求重复数据链路核查。

### 6.2 B：等比例尺度缩小

配置：`crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py`，继承当前SymEOOD主配置。

- real/sim TRAIN都在RResize后，以0.5概率执行 `PortIsotropicShrink`，s均匀取 `[0.5,1.0]`。
- 目前**只有缩小，没有大于1的放大**。不是随机裁剪，也不是改变GT长宽比。
- 图像和OBB使用相同s，零平移，无居中偏移；画布向上取整；累乘scale_factor，保留ori_shape；右/下Pad到1024。
- 测试不执行新增尺度增强，沿用原坐标还原；不增加推理模块。
- 用户曾问是否加放大增强。当前决定先做几何误差分解，没有实现放大版本。

### 6.3 C：光照增强，当前配置不保留为最佳方案

仅real TRAIN，概率0.5，Gamma `[0.7,2.0]`、gain `[0.6,1.2]`、contrast `[0.8,1.2]`；sim管线不加光照。B/C是分别对照，未组合。

C按VAL选epoch16，TEST real全帧中心87.90%、RIoU0.6985、MCML72；sim RIoU0.8594、角度2.9034°。相对B明显差，不据此否定所有光照增强，但不沿当前配置继续叠加。

### 6.4 EOOD＋相同尺度增强的定位

EOOD原方法为论文主基线；EOOD＋尺度是增强对照，用于拆分模型差异和数据增强收益。不能因为用户不想把它当主基线就忽略其结果。

EOOD＋尺度ep24 TEST：real全帧中心96.43%、RIoU0.8230、DFR2.6134、ACI0.9488、TDR100%、MCML8、MRF2.2；sim角度1.4627°、RIoU0.9210、DFR1.4263、ACI0.9586、MCML0。

相较SymEOOD B，EOOD＋尺度在几何精度上更好，SymEOOD B在real中心正确覆盖和MCML上更好。这是取舍，不是全面领先。EOOD增强入口为 `crane_eood_k1_port_day2night_aug_b_v1.py`；保存checkpoint策略显式保留24份以支持16/18/20/22/24扫描，不改变优化过程。

## 7. 已完成的有限TRAIN/VAL几何误差分解

脚本：`crane_project/tools/audit_port_train_val_geometry_v1.py`。

- 复用EOOD B与SymEOOD B所选VAL预测887帧及权重。
- TRAIN固定seed1701，每域32张，共64张；相同图像对比clean与固定0.5缩小，两模型共256次单图推理。
- model.eval、no_grad；不是训练或全量特征审计。缓存绑定配置、权重、源码、样本与GT身份。
- 记录原图中心误差、中心/GT短边、长短边相对误差、有符号log比、RIoU、规范长轴角度，按域/序列/尺度/长宽比/角度分层。
- 近方形框方向存在歧义；把GT中心/尺寸/角度替换进预测得到的IoU增量是描述性反事实，不是可加的因果贡献。

### 7.1 sim VAL512帧

| 指标 | EOOD＋尺度 | SymEOOD B |
|---|---:|---:|
| 输出 | 512 | 512 |
| 平均RIoU | 0.91416 | 0.88570 |
| 平均中心误差px | 1.5253 | 1.6452 |
| 中心/GT短边 | 3.30% | 4.76% |
| 长边相对误差 | 3.25% | 3.87% |
| 短边相对误差 | 2.90% | 4.14% |
| 纯角度RMSE° | 2.1987 | 2.1072 |
| 原协议角度RMSE° | 15.0212 | 2.1072 |

EOOD原协议15°主要来自14帧中心不合法的90°罚项，占平方误差98.16%，不是实际纯方向误差15°。

### 7.2 TRAIN缩小敏感性

- EOOD平均RIoU clean0.9348→half0.8670。
- SymEOOD平均RIoU clean0.9064→half0.8381。
- 两组降幅相近，clean状态已有差距，不能确定缩小强度是几何差距唯一原因。
- VAL没有输入GT短边小于16px的样本，不能从该VAL直接估计所有TEST小目标的尺度下限。

### 7.3 real中的严重错位

SymEOOD B在real_seq07有10帧输出RIoU为零、中心偏移约120–320px。它们会明显拉低整体均值。最终框无法证明是分类排序还是目标候选回归的问题；之后D对照应同时看严重错位修复和普通共同输出精度。

## 8. 中心—尺寸补偿D：实现与验证范围

### 8.1 实验身份与公式

配置：`crane_project/configs/crane_symeood_k1_port_day2night_center_size_d_v1.py`，继承B，仅新增损失配置和独立work_dir。

```text
L_cs = 0.25 × [ mean SmoothL1((pred_xy − GT_xy)/GT_short, beta=0.1)
               + mean SmoothL1(log(pred_long_short/GT_long_short), beta=0.1) ]
L_total = 原有损失 + L_cs
```

- 只在已有SymPOLA正样本计算，沿用全局正样本数归一化，不改变分配规则。
- 使用现有解码/限幅后的OBB；GT detach，长短边排序，无正样本返回连接回归图的零值。
- 新损失无可训练参数、无推理依赖；直接梯度到回归输出前4维，不直接依赖角度或分类输出。
- 梯度经回归头进入FPN和原本可训练的backbone阶段；共享特征仍可能改变分类/角度。
- 日志记录 `loss_center_size_compensation` 和 `center_size_positive_count`。中心和尺寸分项若未单独记录，不能从总损失日志直接恢复其各自训练动态。
- 尺寸项避免等价宽高交换惩罚；共同等比例缩放下保持一致（eps范围以上）。逐坐标SmoothL1不应被宣称为对任意整体旋转完全不变。

### 8.2 相关论文与创新边界

- [DIoU/CIoU，AAAI2020](https://ojs.aaai.org/index.php/AAAI/article/view/6999)：显式中心距离约束。
- [EIoU，Neurocomputing2022](https://www.sciencedirect.com/science/article/pii/S0925231222009018)，[作者预印本](https://arxiv.org/abs/2101.08158)：显式重叠、中心和边长误差。
- [旋转框KLD，NeurIPS2021](https://proceedings.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html)：中心/尺寸/角度联合监督，参数耦合及尺度不变性。
- [GWD，ICML2021](https://proceedings.mlr.press/v139/yang21l.html)：旋转框表示边界与近方形问题。

这些文献支持设计动机，**没有验证当前SmoothL1＋GT短边归一化＋长短边log比＋SymKLD的具体组合和0.25/0.1参数**。当前D不是EIoU原样复现。KLD本来已有中心和尺寸监督，不应把D描述为补上KLD遗漏的监督。

用户认为EIoU值得进行OBB适配，但担心增加角度后出现表示等价、周期和耦合问题。本轮保持SymKLD处理联合几何，暂不加入新角度损失；没有授权或实现完整Rotated EIoU替换。

### 8.3 已做代码检查

- 新loss、真实head集成、状态字典兼容、角度直接梯度为零、GT detach、空正样本、尺度一致性、权重归一化、配置差异及VAL读写测试；相关26项检查曾通过。
- py_compile及git diff --check曾通过。它们证明实现性质，不证明训练收益。
- 短检查脚本在TRAIN的4张real/sim图上，覆盖O2M早期/O2O后期及clean/half视图，真实反传并检查回归/FPN/backbone梯度和clip10，不更新权重。
- 本地没有服务器完整权重/CUDA执行证据。本交接未重新运行GPU、未补做新的审计。
- D已由用户在服务器完成训练和评估，但本轮未提供完整训练日志或短检查报告，不能宣称每一步运行设置均被新窗口重新复核。
- 既有VAL导出身份核验有角点≤0.03px的回退，用于TXT0.01px量化及OpenCV近重合交点数值异常；**没有修改评价RIoU公式**。不能将身份回退误解为修复了所有IoU数值问题。

## 9. D的VAL结果及解释

### 9.1 选权

B已冻结epoch24；D按相同规则选epoch22。D五个候选均通过硬约束：

| D权重 | TDR | Rc_w | ACI | MCML | 角度RMSE° | 软评分 |
|---|---:|---:|---:|---:|---:|---:|
| epoch22，选中 | 1.000 | 0.998 | 0.945 | 1 | 2.20 | 0.9835 |
| epoch24 | 1.000 | 0.997 | 0.947 | 1 | 4.48 | 0.9786 |
| epoch20 | 1.000 | 0.996 | 0.946 | 1 | 4.57 | 0.9780 |
| epoch18 | 1.000 | 0.997 | 0.945 | 1 | 6.08 | 0.9747 |
| epoch16 | 1.000 | 0.996 | 0.943 | 1 | 6.39 | 0.9734 |

### 9.2 VAL完整比较

| 指标 | B ep24 | D ep22 |
|---|---:|---:|
| real输出帧 | 374/375 | 374/375 |
| real输出帧中心命中率% | 96.2567 | 99.4652 |
| real全帧中心正确数 | 360/375 | 372/375 |
| real全帧mean_RIoU | 0.795505 | 0.807446 |
| real输出框平均中心误差px | 9.5746 | 4.6233 |
| real长边相对误差% | 8.4208 | 8.7887 |
| real短边相对误差% | 8.7580 | 8.7085 |
| real纯角度RMSE° | 3.7700 | 3.0073 |
| real RIoU≥0.5帧数 | 364 | 372 |
| sim输出/中心正确 | 512/512 | 512/512 |
| sim全帧mean_RIoU | 0.885695 | 0.882806 |
| sim平均中心误差px | 1.6452 | 1.6504 |
| sim长边相对误差% | 3.8712 | 4.1269 |
| sim短边相对误差% | 4.1388 | 3.8870 |
| sim纯角度RMSE° | 2.1072 | 2.2045 |

- real_seq07 RIoU正确215→223，最长RIoU失败4→1；real_seq14均149正确，RIoU0.88419→0.88750；sim均512正确。
- VAL总886帧输出，共同输出885；D恢复real_seq07_00134，丢失real_seq07_00180，数量相同但不是完全相同帧。
- 中心正确新增12、丢失0；RIoU正确新增9、丢失1，净增8。
- B的10帧严重错位，D修复8帧；00020、00027仍不正确。

利用完整报告在real共同输出373帧中，排除上述10帧历史严重错位，剩363帧：

| 指标 | B | D |
|---|---:|---:|
| 平均中心误差px | 4.3995 | 3.6696 |
| mean_RIoU | 0.81985 | 0.81595 |
| 长边相对误差% | 7.8613 | 8.5043 |
| 短边相对误差% | 7.9287 | 8.4573 |

所以VAL主要证据是中心改善和严重错位修复，**并没有普通框尺寸精度全面提高**。此前同意固定epoch22做TEST，是报告此候选的实际泛化表现，不代表已经满足全部优化目标。

## 10. 最新TEST：D不替换B

下表来自用户服务器终端。B ep24来自已保存附件；D ep22来自本次对话粘贴，尚无D完整TEST JSON/逐帧产物供本地复核。

| TEST指标 | SymEOOD B ep24 | D ep22 | 解释 |
|---|---:|---:|---|
| 总预测框数 | 1431 | 1429 | SymEOOD每帧最多1框 |
| real全帧中心正确率% | 98.85 | 98.16 | D下降0.69个百分点 |
| real mean_RIoU | 0.8169 | 0.8110 | D下降0.0059 |
| real DFR | 2.5103 | 2.5701 | D略差 |
| real ACI | 0.9440 | 0.9412 | D略差 |
| real TDR% | 100 | 100 | 相同 |
| real MCML_max | 4 | 4 | 相同，均通过≤5 |
| real MCML_mean | 3.5 | 3.0 | D改善 |
| real MRF | 2.29 | 1.62 | D改善 |
| sim角度RMSE° | 1.8823 | 2.0589 | D增加0.1766° |
| sim mean_RIoU | 0.8816 | 0.8777 | D下降0.0039 |
| sim DFR | 2.4734 | 3.0598 | D变差 |
| sim ACI | 0.9479 | 0.9449 | D略差 |
| sim中心正确率/TDR% | 100/100 | 100/100 | 相同 |
| sim MCML_max/mean | 0/0 | 0/0 | 相同 |

根据固定real868、sim572、sim全输出及单帧最多1框，由终端反算：

| real输出条件口径 | B | D |
|---|---:|---:|
| 输出帧 | 859/868 | 857/868 |
| 中心正确帧 | 858 | 852 |
| 仅输出帧中心准确率% | 99.8836 | 99.4166 |
| 输出覆盖率% | 98.9631 | 98.7327 |

以上是汇总反算，不是逐帧交集检查。D输出少2、中心正确少6；不能推出“恰好只丢了2帧且没有新增”，也不能确定分别属于seq03还是seq04。

**固定结论：** 当前0.25权重、beta0.1、现有正样本及归一化形式下，D的VAL局部收益未形成TEST整体优势，不满足提高几何精度并保持real正确覆盖的目标。保留B；D是本版负结果/取舍对照，不作为最终改进模型。该结果不证明所有OBB补偿无效，不证明EIoU不可适配，也不证明实现必然有错误。

不能仅凭这次结果断言权重过大、过拟合、梯度冲突或sim样本少是根因。目前每个方案为单次训练，无多种子显著性结论；VAL选出的epoch不同是协议结果，不为TEST改选D其他epoch。

## 11. 关键代码与产物位置

### 11.1 代码入口

所有相对路径均相对于本地/服务器项目根目录：

| 功能 | 文件 |
|---|---|
| SymEOOD主配置 | `crane_project/configs/crane_symeood_k1.py` |
| 当前数据SymEOOD A | `crane_project/configs/crane_symeood_k1_port_day2night_v1.py` |
| B尺度增强 | `crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py` |
| C光照增强 | `crane_project/configs/crane_symeood_k1_port_day2night_aug_c_v1.py` |
| EOOD主/尺度对照 | `crane_project/configs/crane_eood_k1_port_day2night_v1.py`、`crane_eood_k1_port_day2night_aug_b_v1.py` |
| D配置 | `crane_project/configs/crane_symeood_k1_port_day2night_center_size_d_v1.py` |
| 尺度/光照实现 | `mmrotate/datasets/pipelines/port_train_augment.py` |
| D损失 | `mmrotate/models/losses/center_size_compensation.py` |
| D接入 | `mmrotate/models/dense_heads/sym_eood_head.py`、`mmrotate/models/losses/__init__.py` |
| D短检查 | `crane_project/tools/preflight_port_center_size_v1.py` |
| TRAIN/VAL几何分解 | `crane_project/tools/audit_port_train_val_geometry_v1.py` |
| B/D VAL对比 | `crane_project/tools/compare_port_center_size_val_v1.py` |
| 选权/TEST | `crane_project/tools/ckpt_sweep.py`、`crane_project/tools/eval_crane_offline.py` |
| TEST序列统计 | `crane_project/tools/audit_port_test_subsets_v1.py` |
| 相关测试 | `tests/test_port_center_size_compensation.py`、`tests/test_port_train_val_geometry.py`、`tests/test_port_train_augment.py` |

### 11.2 已在本地确认存在的完整报告

- `/Users/mac/Downloads/port_train_val_geometry_v1.json`
- `/Users/mac/Downloads/port_center_size_d_v1_val_compare.json`
- B/C TEST终端附件：`/Users/mac/.codex/attachments/3acfd6bd-c70c-4b04-9f21-8ea9bd67b0a3/已粘贴的文本.txt`

历史对话提过 `test_subset_audit_b_v1.json`、`seq03_fixed_chain_100frames_seq06_v1.json`、`test_subset_legacy_replay_v1.json`，**本次检查Downloads同名路径不存在**。不要假装重新读取了这些文件。需要时先找历史归档/服务器产物；正文已保留其必要历史结论。

### 11.3 服务器B/D产物

```text
work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/
  epoch_24.pth
  val_sweep_port_v1/sweep_results.json
  val_sweep_port_v1/final_test/epoch_24/preds/results.pkl
  val_sweep_port_v1/final_test/epoch_24/preds/Task1_grab/

work_dirs/crane_symeood_k1_port_day2night_center_size_d_v1/
  epoch_22.pth
  val_sweep_port_v1/sweep_results.json
  val_sweep_port_v1/final_test/epoch_22/preds/results.pkl
  val_sweep_port_v1/final_test/epoch_22/preds/Task1_grab/

work_dirs/port_center_size_d_v1_preflight.json
work_dirs/port_train_val_geometry_v1.json
work_dirs/port_center_size_d_v1_val_compare.json
```

上述服务器路径从脚本/用户终端得到，本交接没有连接服务器检查文件是否全部仍在。

VAL报告绑定身份：

| 身份 | B | D |
|---|---|---|
| config SHA256 | `9da972b7010540e12b2501d1c400db13c159381f495f539d6315e7adb498e345` | `848110bd70455b80bbbfa6d44a25f40609db62167e2e66bab927f866430d28cb` |
| checkpoint SHA256 | `8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23` | `5ea9e2e3046a3c1814dd7cdb928e893ba740fa4f637737f2bac1033fbe61f834` |
| VAL PKL SHA256 | `cbfa341c53faf859a4eb96ab62a2470611cde0493896f50fcc7f805f53020725` | `614faa527e47bf034acd4be9d4a9c3ce5dbdcb0c86ccf963cf6401b74f9078ad` |

不要覆盖B/D配置改变已完成实验身份，下一版需独立入口和work-dir。旧压缩代码包 `/tmp/port_center_size_d_v1_code_20260930.tar.gz` 只是历史交付位置，临时目录不保证长期保留。

## 12. 可复用命令与下一步顺序

### 12.1 已完成D TEST命令，仅保留作协议记录

**TEST已完成，不需要再次运行。**

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_port_day2night_center_size_d_v1.py \
  --work-dir work_dirs/crane_symeood_k1_port_day2night_center_size_d_v1 \
  --sweep-dir work_dirs/crane_symeood_k1_port_day2night_center_size_d_v1/val_sweep_port_v1 \
  --final-test-from work_dirs/crane_symeood_k1_port_day2night_center_size_d_v1/val_sweep_port_v1/sweep_results.json \
  --gpu 3
```

### 12.2 若新窗口确需解释D退化：只复用现有TEST框分层

该命令只读现有TXT、计算seq03/seq04/sim_seq09的定位/覆盖/失败区间，不重新推理。它是描述已有结果，不为下一版在TEST选参数。先检查是否已有同样统计，已有则直接复用。

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/audit_port_test_subsets_v1.py \
  --gt-dir crane_project/data/crane_grab_port_day2night_v1/test/annfiles \
  --pred-dir work_dirs/crane_symeood_k1_port_day2night_center_size_d_v1/val_sweep_port_v1/final_test/epoch_22/preds/Task1_grab \
  --out-json work_dirs/test_subset_audit_d_v1.json
```

与B既有分层报告比较即可，脚本当前不是新的完整B/D逐帧配对审计。不要假定已经实现跨模型候选归因。

### 12.3 几何优化接续建议（尚未落实，不是训练授权）

1. 先读B/D的现有TRAIN/VAL报告和必要训练日志，区分严重错位修复与普通框精度，明确下一版只解决哪一个问题。
2. 当前可支持的重点是：中心已有一定改善依据，**尺寸误差尚未改善，sim角度与时序有代价**。不能再次笼统把所有几何项一起增加。
3. 若需要新增检查，限定在TRAIN/VAL，检查与下一方案直接相关的损失分项/归一化/正样本梯度，不自动全量回放、扫描权重或做新的特征审计。
4. 下一实验必须是明确单因素，同模型/数据/增强/预算/选权，保留B作有效对照；需要区分中心和尺寸贡献时，先预登记拆分方案，不把TEST结果用于挑系数。
5. 当前不自动提高D权重、不加放大尺度、不加新角度项、不改SymPOLA、不重新划分数据，也不默认改成完整Rotated EIoU。需要先解释机制与风险，再由用户确认具体方向。
6. 如果下一版无法保持real正确覆盖，也没有尺寸/角度证据，就如实保留B，不继续用局部指标包装全面收益。

论文可写：SymEOOD与尺度增强在当前港口划分上改善real覆盖和连续性，存在跨域几何精度取舍；D证明显式补偿并非自动获得全面收益。尚不能写：已经提出可靠的新OBB损失、已经保证深度估计不受影响、全面优于EOOD、所有蒸馏路线均被穷尽或学生已达到表达容量上限。

## 13. 更早研究的必要背景：防止走回原路

这部分来自此前对话与已有归档，不要求新窗口重新逐项审计。

- DINO→SymEOOD特征蒸馏V1从头训练损失较多K1输出；V2 K1初始化同预算仅新增1帧中心命中、连续缺测未改善。
- V3比较FPN蒸馏梯度阻断/允许，随后发现掩码问题；V4修正掩码并做A/C正式对照，没有形成稳定核心收益。不能将V3实现问题直接扩写成所有负结果都无效。
- V5对象—邻近背景关系蒸馏24epoch，关系项只更新分类适配器、保护FPN几何路径。真实中心正确数没有增加，最长无输出39→64、sim几何下降；complete产物核查已完成。停止扫描该关系权重、背景环和训练轮数。
- V5 source VAL候选诊断：A成功735/738、排序失败2、阈值失败1；C成功729、排序失败3、阈值失败6，输出少6。实际SymEOOD路径无RPN/NMS，不可照搬两阶段诊断。
- 直接GT候选选择监督是后续尝试，**不是蒸馏**。4张预检＋128张分层source补查共132张未激活margin0.1的新损失，未验证真实新增梯度及活动样本冲突率；不因此证明全体训练都不激活或容量已饱和。
- Native DINO完整检测器与冻结DINOv2缓存教师身份不同。旧source VAL小范围对照677/738对735/738不能证明DINOv2特征无价值，也不是当前新数据几何结论。
- learned S7等历史候选排序有27条路线记录，负结果不能换名字重复；仅有配置且work-dir删除，按用户说明视为做过的机制线索，不能因缺失产物自动宣称从未尝试。
- 历史MCML=6等教师指标必须保留来源限制，不当作学生应达的确定上限。公开困难片段、native检测器、蒸馏缓存、不同输入分辨率不能混为同一“DINO”。

当前优化中心已经转到**新数据集、SymEOOD＋尺度B及其几何精度**，不以扩大DINO系统或再建蒸馏损失作为默认下一步。

## 14. 可直接粘贴到新对话的接续请求

> 请先读取 `/Users/mac/Documents/paper/symEOOD/docs/geometry_precision_handoff_20261001.md`，继承当前SymEOOD几何精度研究。当前有效方案为含seq06的新数据集上的SymEOOD＋等比例尺度缩小B，VAL选epoch24；中心—尺寸补偿D选epoch22，VAL改善real错位和中心覆盖，但TEST中心覆盖、RIoU及sim角度/时序大多略退化，因此不替换B。目标是保持real正确覆盖和连续性，同时提高尺寸、定位及sim角度精度，暂不考虑DINO。先复用已有TRAIN/VAL结果，提出一项范围有限、机制明确的下一步设计；不要重复完整审计，不用TEST调参，不立即改代码或启动训练。保留两种中心指标分母、旋转框表示等价和深度坐标还原约束，修改记录集中更新现有文档。


## 15. 接续静态核对与下一项设计（2026-10-01，尚未实施）

### 15.1 本轮边界与已核对事实

本轮读取原项目交接、B配置、SymKLDLoss及底层算子、主头正样本损失与推理还原、D损失、尺度增强实现、选权脚本相关规则，并读取Downloads中的两份完整TRAIN/VAL JSON。没有连接服务器、推理、训练、改代码或读取新增TEST产物。当前仍保留B ep24；D ep22只保留为对照负结果。当前TEST已多次暴露，不能声称研究全程使用未接触测试集。

B、D本地配置SHA256与第11节报告身份一致；权重与服务器运行日志没有在本轮重新验证。报告复核支持第7、9节结论：sim的纯角度并非相对EOOD最明显差距，中心/尺寸仍有缺口；D的real普通框尺寸没有全面改善。B VAL real中心正确360/375、输出374/375；正文应同时报告输出帧中心命中360/374=96.2567%、输出覆盖374/375=99.7333%、全帧正确覆盖360/375=96.0%。

代码确认：SymKLD的原始距离可分成 C+S，其中

```text
C = 0.5 * delta^T (Sigma_pred^-1 + Sigma_gt^-1) delta
S = 0.5 * [tr(Sigma_gt^-1 Sigma_pred)
         + tr(Sigma_pred^-1 Sigma_gt) - 4]
f(x) = min(sqrt(1+x)-1, 10)
L_bbox = 2 * weighted_reduce(f(C+S), global_positive_count)
```

上述分解为正常有限值、数值保护未激活时的表达。实际算子另有宽高>=1、行列式下限、逆矩阵限幅及raw<=1e4保护。原损失上限10在raw>120后梯度为零；源码“sqrt对>1000比log压制更强”的注释不能作为数学结论，sqrt本身的导数比log衰减慢，上限截断是另一机制。尚无训练正样本饱和比例，不能由此认定B训练存在大量零梯度。

为判断是否值得仅改分项压缩，本轮只用报告中B的原图最终框，以双精度解析2x2协方差计算 C、S。未执行新推理。这不是训练正样本梯度测量，亦未重放实际解码/限幅；用于有限的机制筛选：

| VAL域 | 输出数 | C中位数 | S中位数 | 分项压缩相对原式的形状项导数倍率中位数 | 倍率>1.1帧数 |
|---|---:|---:|---:|---:|---:|
| real | 374 | 0.006557 | 0.032436 | 1.00310 | 10 |
| sim | 512 | 0.002755 | 0.011865 | 1.00135 | 0 |

倍率是 sqrt((1+C+S)/(1+S))，指对S这一标量项的偏导系数，不是回归参数总梯度倍率。real最大3.89457，sim最大1.03711。由此推断：只将f(C+S)改为f(C)+f(S)，对现有sim普通框直接影响很小，不作为本轮推荐；不能由最终框统计排除其训练早期效果。

### 15.2 唯一推荐：B＋同中心协方差形状补偿E（暂定实验代号）

目标优先是尺寸精度，sim纯角度是共同观察终点；不承诺一项改动同时解决所有定位问题。保留B的全部联合几何监督和尺度增强，新增正样本上的形状项：

```text
L_E = L_B + 0.25 * weighted_reduce(f(S), global_positive_count)
```

S使用上式协方差迹项，不读取预测/GT中心；沿用现有高斯映射、数值保护、正样本与权重。GT detach，空正样本返回连接回归图的零值。若后续实施，独立配置/目录，不覆盖B或D。

**机制与推断：** 原中心马氏项C的逆协方差依赖预测尺寸和方向；在中心偏差存在时，改变尺寸也可能降低该项，联合最优不必等于尺寸单项最优。额外S把尺寸与方向拉向GT协方差，不再增加D的中心项，也不把长短边和角度拆成三个独立回归目标。在常规小误差区约增加12.5%的形状项监督强度；0.25是预登记的保守起点，不是经实验验证的最优值，也不是与D等梯度强度的匹配。

协方差对(w,h,theta)与(h,w,theta+pi/2)及theta+pi表示等价；近方形框的方向约束自然减弱，不强迫一个不可靠长轴角。共同等比例缩放下C、S理论保持不变，实际下限/eps附近需检查。新增项对解码框中心的直接偏导为零，但共享回归塔、FPN、backbone以及依赖预测的分配仍可能改变中心与分类。它不能保证real正确覆盖、连续性或深度精度。

**创新边界：** 这是现有对称高斯几何损失的形状加权适配与对照，非新颖性已确立的原创损失。它与D不同在于移除新增中心监督、用联合协方差约束尺寸/方向；不是完整Rotated EIoU，也不恢复蒸馏或候选排序。

### 15.3 对照与必要缺口（后续授权后执行）

- 主要对照为已冻结B ep24，实验E从相同ImageNet初始化训练；不从B续训。数据含seq06、B增强、seed0、24epoch、优化器/学习率、SymNFL/SymPOLA和辅助分支、推理阈值及选权规则均不变。只改变上述新增项；D作历史取舍对照，不据其TEST下降调整系数。第一轮不扫权重、不叠加放大、中心或独立角度损失。
- 实施前仅补与E有关的检查：表示等价、等比例一致性、GT detach/空正样本/数值有限、解码框中心直接梯度为零且尺寸/非方形角度有梯度、全局归一化，以及head集成。复用已有4张TRAIN clean/half及O2M/O2O短检查入口，分别记录原回归与E项的损失、梯度范数及夹断比例，不更新参数。B权重仅用于此短检查，不改变正式初始化。小样本夹角仅作风险线索，不证明全TRAIN冲突率。若B权重本地不可用，只给服务器检查指令，不连接服务器。
- 若实施性质或数值稳定性检查失败，先修复；若新增项几乎无有效梯度或异常主导原回归梯度，暂停正式训练并报告，不自动改系数或扩大诊断。
- E仍扫描16/18/20/22/24，按原契约选权，披露是否触发原脚本fallback；不因几何报告另选epoch。只对固定所选权重复用现有VAL比较工具的统计方法。
- 是否替换B需另过VAL目标门槛：real输出数>=374、全帧中心正确数>=360、最长无输出与最长RIoU失败均不劣于B；sim输出/中心正确维持512。加权Rc约束不能代替real单域门槛。比较采用同序列边界及相同采样缺口处理。
- 在这些门槛下，real与sim共同输出框长/短边相对误差均须下降，sim纯角度RMSE须下降（基准2.1072度），共同输出中心误差不增、全帧RIoU不降；同时报告完整输出框统计、p90、有符号尺寸偏差及输出增减。real普通框与历史10帧严重错位分别描述，不能用修复错位掩盖普通框尺寸退化。门槛严格，无联合收益就保留B，不用单次微小差异宣称显著性。
- TRAIN64张clean/half缓存只作预先误差参照；无需重跑EOOD或完整审计。单次seed0只能得到探索性结论，有稳定收益后再决定是否需要多种子验证。
- 设计、系数与VAL权重冻结前不增加TEST检查；后续TEST仅在明确授权后报告冻结方案，不能调参或重选权重。已有多次TEST暴露需在论文评估限制中披露。

图像增强、原图框还原和内参契约完全沿用B；不引入裁剪、平移或不等比例变换。新增监督不等于预测尺寸/深度可靠性已得到保证，测量仍需独立真值验证。

**当前状态：设计完成，未实施、未训练，尚无E收益证据。** 本节集中记录本轮新事实、机制推断与待验证事项，不更改历史实验身份。


### 15.4 E相关文献检索与证据边界（2026-10-01）

用户认可E作为有限形状加权实验，并强调不能把数学机制认作当前退化根因。本轮据此检索原始研究（Exa三组成功搜索，共请求22条候选，候选含重复；另通过网页工具核对会议/作者原文），未扩展完整审计，未改代码或启动训练。以下是与设计直接有关的证据，非穷尽的新颖性检索。

| 原始研究 | 已读证据与可支持内容 | 对E的限制 |
|---|---|---|
| [Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/98f13708210194c475687be6106a3b84-Paper.pdf) | 第3节式8/9区分中心与协方差项，式13–15分析尺度、尺寸和角度梯度；支持高斯几何及角度敏感性随长宽比变化。原文也把参数耦合解释为有效优化机制。 | 未验证本项目增加0.25*f(S)，不能将所有中心—形状耦合说成缺陷，也不能声称论文已证明B偏小的根因。 |
| [Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss，ICML 2021](https://proceedings.mlr.press/v139/yang21l.html) | 第4节及原文PDF讨论高斯映射、宽高交换、角度周期及近方形问题，为表示选择提供基础。 | GWD与本项目对称KLD是不同度量；损失表示的平滑不能直接推导本模型推理角度完全连续。 |
| [The KFIoU Loss for Rotated Object Detection](https://arxiv.org/html/2201.12558)（[版本与发表信息](https://arxiv.org/abs/2201.12558)） | 第4.1/4.2节：保留中心损失，另用只依赖协方差的高斯乘积重叠项；表1还比较以KLD中心项替代常规中心项。支持“中心定位和尺寸—方向形状可以分别处理”的已有先例。 | KFIoU是协方差重叠近似，本项目S是对称KL迹项；该方法及其实验不是保留整个SymKLD再额外增加形状项，不能称E为KFIoU复现。 |
| [Rethinking Boundary Discontinuity Problem for Oriented Object Detection，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Rethinking_Boundary_Discontinuity_Problem_for_Oriented_Object_Detection_CVPR_2024_paper.pdf)（[作者全文](https://arxiv.org/html/2305.10061)） | 摘要及方法分析指出，仅在loss计算中平滑角度，仍可能留下回归输出的边界不连续；提出角度双优化编码。 | E沿用原OBB输出和解码，不解决该文针对的编码问题。该论文不能用来认定当前sim角度误差就是边界问题。 |
| [GauCho: Gaussian Distributions with Cholesky Decomposition for Oriented Object Detection，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Marques_GauCho_Gaussian_Distributions_with_Cholesky_Decomposition_for_Oriented_Object_Detection_CVPR_2025_paper.pdf) | 摘要、第1/2节区分高斯loss与直接高斯输出；直接回归Cholesky参数，讨论正方形映射丢失方向信息及推理边界问题。 | 属于回归头/表示改动，超出E单因素范围，本轮不引入。它提醒不能把近方形角度约束减弱宣称为所有角度信息均已正确保留。 |

**数学推导（本项目推导，不是文献收益结论）：** 在保护未激活时，当前raw=C+S等于 KL(Np||Ng)+KL(Ng||Np)，即两个方向KL之和（Jeffreys divergence；不是两者平均）。将两个高斯中心设成相同值后，两个方向的中心项为零，log-det项抵消，所得和正是S。因此E的新增项是同中心对称KL形状正则，而非新定义的统计距离。

对S的标量偏导，新增项相对原项的比例为

```text
[0.25*f'(S)] / [2*f'(C+S)]
  = 0.125 * sqrt((1+C+S)/(1+S))
```

仅在有限值、非夹断区成立；小误差约12.5%。这不是网络参数总梯度强度，实际还含C对尺寸/方向的导数及共享参数链路。短TRAIN检查仍然必要。对于严格正方形w=h，高斯协方差不含theta；不同旋转的正方形可能有不同RIoU却映射为相同协方差。测试应明确这是表示局限，不能把此时角度梯度为零判成实现错误，也不能把它当成深度或机械轴角已可靠的证据。

**检索结论：** 已找到支持几何表示、参数耦合和中心/形状分开处理的相关研究；在本轮核对的原始研究中，未找到对“保留2*f(C+S)，额外增加0.25*f(S)”的直接验证。这不证明该公式从未出现，也不确立原创性。E仍保持第15.2/15.3节固定设计和评价门槛，其定位是检验形状监督相对权重的适配假设。即使B/E改善，也不能单独证明D失败由中心补偿造成，或证明形状监督不足是唯一根因。

**状态：文献依据已补充；E尚未实施，无新TRAIN/VAL/TEST结果。** 后续按有限实现检查→短TRAIN检查→正式同预算实验顺序推进；正式训练由用户在服务器执行，本地不连接服务器。文献检索不触发TEST调参，不放宽已登记的覆盖与几何条件。


## 16. D详细机制检查：本地实现、服务器待运行（2026-10-01）

### 16.1 授权与范围

用户要求更仔细检查D，并明确本地没有正式B/D日志、权重和D完整VAL框；授权“在项目文件夹下修改代码，我上传服务器运行”。本轮只新增诊断工具与相关测试，不改B/D损失、配置、数据、模型或推理行为；不实施E、不训练、不连接服务器、不读新增TEST产物。用户运行诊断后，先解释D的证据，再决定E是否需要调整。原先已暴露的TEST仍只保留历史结论，不参与参数或权重选择。

新增文件：

- `crane_project/tools/diagnose_port_center_size_d_v1.py`
- `tests/test_port_center_size_d_diagnosis.py`

### 16.2 细查后的静态事实与限定

1. D配置解析后只增加CenterSizeCompensationLoss及独立work_dir，继承B的数据、增强、初始化、优化器及24epoch预算；历史B/D身份不变。新增项只用实际SymPOLA正样本，GT detach，复用各层的权重与同一分母；公式重建与真实head输出需要逐层数值/梯度一致。
2. 先前“全局正样本数”需要精确理解：get_targets在当前GPU/rank的batch内跨层汇总，逐图使用max(npos,1)，当前路径没有跨rank all-reduce。D与B的主分类/回归共用这一约定；不能单凭此称D引入归一化错误。诊断单卡使用batch2（一张real、一张sim），不等于正式双卡DDP的完整更新，但可检查同一卡上的实际归一化与梯度路径。
3. 当前主头实际retina_reg/retina_cls是直接3x3卷积；配置stacked_convs=4并没有在该实现中生成四层回归塔。D的中心/尺寸直接梯度作用于不同输出通道，其主回归卷积梯度内积可能为零；这不等于共享FPN/backbone没有梯度取舍。该结构由B/D共用，本轮不改变它。
4. SymNFL几何权重使用detach后的KLD，分配也detach。新增D没有直接分类或角度梯度，但预测变化仍会改变后续分配与分类权重；共享FPN/backbone也可能受影响。
5. 现有preflight只记录D合计、若干梯度范数与clip前范数，无法还原中心/尺寸各自的训练历史；没有原preflight报告也不能补称其已运行通过。有限值日志不能排除原head曾把非有限loss置零。诊断必须保留这些不可识别限制。
6. 本地D VAL JSON保留完整分层汇总和少量正确性变化行，未保存所有D逐帧框。本轮不能据此宣称完整B/D逐帧配对已重新完成；新脚本会在服务器直接读取原始选权PKL/TXT补足，不重新推理。

### 16.3 已有TRAIN缓存的有限补充计算（事实，非网络根因）

复用`/Users/mac/Downloads/port_train_val_geometry_v1.json`中SymEOOD B的TRAIN64张clean/half最终输出框，按报告input_gt_short_px/GT短边将框映射到模型输入尺度，用实际D公式与SymKLDLoss计算。每域32张；没有加载权重或新推理。

| 视图/域 | D中心损失 | D尺寸损失 | 原SymKLD（权重2） | D合计/原损失 |
|---|---:|---:|---:|---:|
| clean real | 0.001288 | 0.002046 | 0.014286 | 0.233 |
| clean sim | 0.001544 | 0.000759 | 0.012649 | 0.182 |
| half real | 0.004631 | 0.003489 | 0.033317 | 0.244 |
| half sim | 0.008156 | 0.005525 | 0.047682 | 0.287 |

这些是B最终输出上的反事实loss值，非训练正样本、非梯度范数，也不是D快照。不能据它调整0.25或断言中心主导全训练；只说明损失值强度及两项占比确实随域/尺度变化，不能只看系数大小判断影响。

### 16.4 新工具实际检查内容

**原产物与身份：** B固定epoch24、D固定epoch22；读取各自sweep_results.json、checkpoint SHA、VAL PKL SHA、VAL标注集合SHA，检查PKL/TXT对应，要求887帧及固定三个VAL序列。工具不调用会写manifest的旧检查入口、不扫描或重选其他epoch。分别读取两目录现有`*.log.json`并记录SHA，按epoch统计loss、正样本数、lr、grad_norm、非有限/坏行；多日志不擅自拼成一条完整训练轨迹。checkpoint中保存的config、epoch/iter/seed也记录，并比较关键解析配置；缺失或差异保持“待复核”。

**VAL结果：** 复用完整框，保留两种中心分母和输出覆盖，给出共同输出逐帧变化、real历史RIoU=0错位组与其他共同输出组、各域/尺度/序列分层、尺寸有符号偏差和p90、纯角度与罚项角度。最长无输出和RIoU失败分开记录，序列边界与编号缺口处重置。共同输出不能掩盖新漏检；这些按帧统计不等于真实时间控制可靠性。

**TRAIN真实反传：** seed1701，每域8张，共16张，real覆盖五个TRAIN序列。相同图像分别clean/固定0.5缩小，不随机翻转；在B/D冻结快照上各检查O2M早期、top-k过渡中点和O2O后期，共96个batch2前向、192个图像视图。不执行optimizer，不保存修改后权重；模型参数前后SHA必须相同。B快照中的D项仅用于反事实诊断，其实际完整B目标不包含D。

每个batch拆开D中心/尺寸项并核对其和等于实际输出；比较它们与原SymKLD、主分类、其他辅助损失及完整B目标的梯度范数/方向。记录主分类/回归完整卷积参数，以及一个明确标出的FPN参数和一个backbone参数探针；后两者不是整个FPN/backbone梯度。记录各域的解码框无量纲直接梯度，但共享参数梯度仍为混合batch，不能冒称逐域FPN梯度。

同时记录实际正样本数与每层分母、B/D相同图像/阶段的正样本交集、归一化中心残差与log边长残差、C/S组成、SmoothL1线性区计数、解码中心/边长限幅、delta宽高限幅、协方差行列式/逆矩阵保护、KLD raw与loss上限、完整实际目标clip10前后范数及裁剪倍率。小样本负夹角不是全TRAIN冲突率，冻结后期快照强制早期分配不等于恢复早期训练历史。

**报告保存：** 完整JSON输出不可覆盖旧文件；GPU开始前先保存`.artifacts.json`，每个完成batch追加`.progress.jsonl`。异常停止时保留已完成证据。正常结束状态为`DIAGNOSIS_COMPLETE_REVIEW_REQUIRED`，不自动宣布根因或授权E训练。

### 16.5 本地验证与服务器运行

本地`/opt/anaconda3/envs/mmrot/bin/python`、torch1.8.0、mmcv1.7.2 CPU：新增9项测试通过，包含真实head两种分配的反传、分项loss/梯度等价、空正样本、限幅统计、日志非有限/坏行处理、VAL丢输出/缺口统计、checkpoint配置差异识别及三阶段运行流程的CPU I/O夹具；配置核对通过，语法与空白检查通过。流程夹具使用小型共享特征和伪造I/O，只用于验证代码流程，不证明ResNet/CUDA/服务器实际权重运行通过。本地未重复全套历史26项检查，也未执行CUDA整网。

上传新增脚本（测试可选）到服务器同名相对路径；使用原B/D产物，不需要重新训练或导出VAL。运行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/diagnose_port_center_size_d_v1.py \
  --gpu 0 \
  --per-domain 8 \
  --out-json work_dirs/port_center_size_d_diagnosis_v1.json
```

物理第四张卡映射为进程内gpu0。默认B/D sweep路径与第11节相同。若只读已有产物且不做GPU反传，可另用`--artifact-only`并给不同的新输出名；不要求先后重复执行两种模式。常规完成后只需带回主JSON（包含完整VAL与TRAIN）；异常时带回错误信息和已保存sidecar，不删除旧报告掩盖失败，也不自行增加样本/扫描系数/检查TEST。

**待验证：** 服务器实际日志、checkpoint配置、TRAIN正样本梯度与VAL完整配对；D退化根因仍未确定。完成后优先回答是实现/数值异常、两项强度不均、共享梯度取舍或未解释的泛化波动；证据不足时明确保留未知，不把任一小样本相关性写成因果。E设计暂不改，等待D检查结果。


### 16.6 服务器采样入口失败修复（2026-10-01，fix1）

**事实：** 用户运行诊断后，VAL原产物读取与TRAIN两视图的数据集构建已完成，随后在pick_indices处报`AttributeError: 'ConcatDataset' object has no attribute 'data_infos'`。失败发生在加载B/D模型、TRAIN反传之前，不是D训练失败或D损失错误。旧脚本把ConcatDataset传入只接受平面data_infos的采样器；此前CPU流程夹具直接暴露data_infos，未覆盖真实包装层，是诊断工具的适配缺陷。

**修复：** 新增ordered_data_infos递归按ConcatDataset子集顺序展开元数据，校验数量等于全局索引长度，给原采样器传入只读平面视图。保持原seed1701、各序列分层方法和实际dataset全局索引；检查clean/half元数据顺序一致。没有修改历史几何脚本、模型、损失或数据，也不改变B/D实验身份。

**验证：** 新增/更新测试共16项通过；使用真实MMDetection ConcatDataset检查嵌套包装、序列覆盖、全局索引与旧平面采样的结果一致，并把CPU运行流程夹具改成真实包装层。在原项目目录还完成实际TRAIN数据集元数据检查：real1810＋sim748=2558，两视图顺序一致；选中real8＋sim8，real_seq01/05/06各2张、12/13各1张，sim_seq08共8张。工作树不含本地数据，所以此数据检查使用原项目的只读数据目录；未读取服务器、未运行新推理或训练。语法/空白检查通过；CUDA整网反传仍待用户运行。

**接续而非删除报告：** 新增`--resume-empty-progress`，仅允许主JSON尚不存在、原.artifacts.json和.progress.jsonl均存在且没有已完成TRAIN batch时接续。复核原冻结选权、配置/模型源码、checkpoint/PKL/VAL标注SHA不变，保留原artifact字节和原源码SHA，在新主报告另记修复后诊断源码与被复用artifact的SHA；只允许诊断工具本身更新，不能借此修改历史模型/配置身份。复用原VAL配对，不重复推理或选权；日志重新只读采集。若已有TRAIN进度或身份变化则明确拒绝，不覆盖旧输出。

本次采样入口失败对应空progress，可上传fix1脚本后在相同项目目录运行：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/diagnose_port_center_size_d_v1.py \
  --gpu 0 \
  --per-domain 8 \
  --resume-empty-progress \
  --out-json work_dirs/port_center_size_d_diagnosis_v1.json
```

保留既有`work_dirs/port_center_size_d_diagnosis_v1.artifacts.json`和`.progress.jsonl`，无需删除或重训。上传代码包为`port_center_size_d_diagnosis_v1_fix1_20261001.tar.gz`，只含诊断脚本及测试；旧代码包保留。此修复不产生D退化根因结论，E仍等待诊断结果。

## 17. D服务器诊断分析（2026-10-01，TRAIN/VAL；E尚未实施）

### 17.1 实验身份与完成情况（事实）

输入：用户回传终端文本、`/Users/mac/Downloads/port_center_size_d_diagnosis_v1.json`，JSON SHA256为`4062ce140b77a5e80b4d208c7cb71ceb1a7bcc84d413fd9bab25a153a7c3c487`。本轮未连接服务器、未运行新推理/训练/TEST、未重选权重；本地仅核对源码、重算JSON和运行一个CPU计算图复现。

首轮ConcatDataset采样错误发生于模型加载和反传之前，是第16.6节工具适配缺陷，不是D训练失败。fix1以`--resume-empty-progress`完成接续，终态`DIAGNOSIS_COMPLETE_REVIEW_REQUIRED`。完整结果包含B/D各887条VAL记录、96个batch2反传探针、12组汇总和48条正样本分配交集。problems为空；optimizer_steps=0、training_epochs=0，两模型参数前后SHA相同。GTX1080、torch1.13.1+cu117，峰值已分配显存约2419MiB。

固定B epoch24/iter15360、D epoch22/iter14080，seed均0；checkpoint保存配置合同均MATCH，解析配置仅D补偿及work_dir不同。8项源码SHA均与本地原项目当前文件一致。12组loss均值、各域中心分母和平均RIoU经原始行重算一致。没有扫描其他epoch。

TRAIN仅16张独立图像（每域8张，seed1701），两种等比例尺度、三个强制分配阶段反复使用，形成192个图像视图，不能当作192个独立样本。后期冻结快照强制早期分配不代表早期训练历史。过渡中点逐图计数更新使两图正样本为5+4、合计9；早期18、后期2，不应假定过渡两图均top-k5。

### 17.2 实现、数值与梯度（事实及限制）

1. 96次探针的D中心＋尺寸分项均通过与实际补偿loss等价断言；loss及实际完整目标参数梯度有限。D中心只直接作用xy，尺寸只直接作用wh，直接角度梯度、主分类卷积D梯度均为0；共享特征、后续分配和几何调制仍能间接改变角度/分类。
2. 正样本解码中心/边长限幅、delta宽高限幅、协方差行列式/逆矩阵保护及KLD raw/loss上限计数均0，不能据此把当前抽样退化归咎截断。仅1/96次完整实际目标触发clip10：D half强制早期第2个batch，裁剪前10.9491、倍率0.9133。此结论不覆盖历史早期或全部TRAIN。
3. D后期主回归卷积的中心/KLD梯度范数比中位数clean0.186、half0.188，尺寸/KLD为0.091、0.065，两项合计/KLD为0.243、0.203。FPN单参数探针合计比为0.236、0.207。补偿有效且不可忽略；但clean FPN尺寸比0.165大于中心0.141，不能说中心在所有共享位置主导。
4. D合计与原KLD的余弦：主回归卷积96/96有效、FPN探针94/96有效，均正，未见普遍反向抵消。共享探针中心/尺寸及补偿/分类有局部负余弦，部分接近0；不能只据负号声称强冲突或因果根因。FPN/backbone各只测一个参数，零范数或None不等于整个模块没有梯度；混合batch梯度也不能外推逐域共享层结论。
5. 后期TRAIN正样本D相较B，尺寸分项均值clean降低17.5%、half降低34.4%，中心项却提高11.7%、18.1%，原KLD也较高。但B/D后期正样本Jaccard均值0.667，且选中epoch不同，不是同候选因果对照。仅能说抽样尺寸代理loss降低未直接转化成VAL联合收益。
6. B/D完整日志各288条TRAIN记录、24epoch；另一个D日志无TRAIN记录，单独保留、不拼接。记录无非有限值/坏行。epoch1均出现共同启动大梯度（最大约4.61e5），12/12日志值大于10；epoch2均6/12，之后无记录值大于10。interval50日志不能换算实际step裁剪比例。不能把共同启动现象归咎D；历史head会把非有限loss置零，因此有限日志不证明历史全部原始loss有限。D补偿均值从epoch1约0.09867降至epoch24约0.00304，说明监督生效；两种总目标不同，不能直接比较总loss优劣。

### 17.3 直接梯度探针缺口（事实，非模型缺陷）

报告96条`direct_decoded_box_gradient.symkld`全部为0，逐域`dimensionless_norms.symkld`也为0、相对KLD夹角为None。这些字段不能用于逐域D/KLD强度或方向结论。

源码原因：head原KLD使用`pos_pred_bboxes = decoded_pred_bboxes[pos_inds]`（约705–718行）；D使用另一次`decoded_pred_bboxes[pos_inds]`（约791行），两者是同一父张量的兄弟切片节点。诊断在D节点上调用`autograd.grad(bbox, pos, allow_unused=True)`（约524–528行）；原KLD不依赖该D节点，返回None被grad_norm记为0。数值相同不等于计算图节点相同。CPU以真实SymKLDLoss复现：loss0.119655，实际KLD切片与父张量梯度范数均0.527318，D兄弟切片梯度为None。

参数级KLD梯度非零，所以主回归/FPN参数对照仍有效，D自身直接梯度也有效。`problems=[]`没有覆盖这个探针错误，不代表所有字段已验证。原JSON不改，不需要重跑96batch或VAL。下一轮若使用逐域直接对照，应针对同一实际正样本图节点，或共同父张量求梯度后筛选正样本；明确区分“无图依赖”和“真实零梯度”，增加最小回归验证。这里只记录建议，未改代码。

### 17.4 VAL配对：收益和代价（事实，描述性）

中心命中仅统计输出帧，覆盖另报：

| 域/模型 | 输出中心命中率 | 输出覆盖率 | 全帧中心正确覆盖 | 全帧平均RIoU |
|---|---:|---:|---:|---:|
| real B | 360/374=96.2567% | 374/375=99.7333% | 360/375=96.0000% | 0.795505 |
| real D | 372/374=99.4652% | 374/375=99.7333% | 372/375=99.2000% | 0.807446 |
| sim B | 512/512=100% | 512/512=100% | 512/512=100% | 0.885695 |
| sim D | 512/512=100% | 512/512=100% | 512/512=100% | 0.882806 |

D修复B历史10帧RIoU=0严重错位中的8帧，00020/00027仍失败。real_seq07最长RIoU失败段4→1帧；最长无输出仍1帧，补回B漏出的00134、新增漏出00180。可以说RIoU失败连续段改善，不能说全帧输出或中心全部正确，也不能等同真实控制连续性。

全real纯角度RMSE3.7700→3.0073度，但普通框没有同步收益。按既定B严重错位名单，从373个共同输出排除10帧，得到363个普通共同输出：

| 普通real共同输出指标 | B | D |
|---|---:|---:|
| 平均中心误差(px) | 4.3995 | 3.6696 |
| 平均RIoU | 0.819847 | 0.815949 |
| 长边平均绝对相对误差 | 7.8613% | 8.5043% |
| 短边平均绝对相对误差 | 7.9287% | 8.4573% |
| 纯角度RMSE(度) | 2.2105 | 2.4397 |

363帧中中心改善215帧，RIoU176帧、长边168帧、短边165帧、角度174帧。分组仅描述固定模型取舍，不更改选权条件或宣称显著性。real_seq14也有长短边误差和角度RMSE上升，尽管RIoU略升。

sim512帧全部共同输出：中心1.6452→1.6504px；长边误差3.8712%→4.1269%，短边4.1388%→3.8870%；纯角度RMSE2.1072→2.2045度，中位角度误差1.1076→1.3526度、p90为3.4121→3.7733度。280/512帧角度变差，说明代价不只集中于单个极端帧；不是跨种子稳定性结论。

全real输出框长/短边signed log ratio均值B约-0.04755/-0.05762、D约-0.05971/-0.06966，整体负偏差未消除。均值log比不是平均百分比误差，也不能证明“中心项驱动缩框”。保持等比例变换/原图还原并不保证深度精度；本报告没有独立深度GT验证。

### 17.5 结论与下一步（推断和待验证）

**事实结论：** D在检查范围内正确接入、有效反传，未见限幅或普遍补偿—KLD反向抵消解释退化。其结果是real严重错位/中心改善伴随普通尺寸及方向代价；未形成VAL联合优势。保留B epoch24，D epoch22作为本版负结果/取舍对照。

**机制推断：** 中心/尺寸相对梯度随尺度、参数位置变化，共享特征和预测相关分配存在取舍；独立边长补偿不直接增强尺寸—方向联合匹配。这与观察相容，但少量样本、冻结快照、不同epoch和不同分配不能确定因果。不能声称已证明中心补偿导致失败、形状梯度缺失、原KLD耦合错误或0.25过大/过小。

**建议：** 不再追加完整D审计、扫描epoch/系数或重做VAL。下一轮局部实现E时先修正17.3探针，通过最小CPU图依赖检查，并仅用有限TRAIN批次补关键缺口。E维持第15节：原B目标不变，新增0.25*f(S)，不加D中心项、不改分配/教师/推理/等比例变换/原图还原/深度链路。检验的是联合协方差形状监督的相对权重是否适合本项目，不是已经确定的根因修复或新OBB损失。

有限TRAIN检查包括S数值/参数梯度、xy无直接新增梯度、wh和非方形框角度梯度、近方形敏感性、保护和裁剪。通过后同初始化/seed/预算只作一次正式B/E对照，沿用第15.3节VAL选权和预登记覆盖、连续性、联合几何门槛；分别报告普通real和严重错位组，不用修复错位掩盖普通框退化，不在看到结果后放宽门槛。单次微小差异不称稳定/显著收益。本轮未修改诊断/模型代码，未自动授权或启动E训练。

**TEST披露：** 当前TEST已经多次暴露，不能当作未触碰的独立确认集；本轮只使用TRAIN/VAL，不据TEST调整E系数、评价条件或重选权重。以后若需独立泛化结论，应事先冻结方案并使用尚未暴露的独立序列/数据；当前无需因此追加采集或连接服务器。

## 18. 根据D诊断开展的定向文献检索与E设计复议（2026-10-01；未实施）

### 18.1 用户新要求与当前状态

用户要求先依据现有诊断和代码查论文，再讨论E实验。因而第15节的`0.25*f(S)`记为原候选E0，不再把“必须维持该公式”作为下一步前提；第17.5节保持E公式的建议是上一轮意见，本节为最新状态。B epoch24仍是有效方案、D epoch22仍是取舍对照，数据/选权/覆盖和联合几何门槛不随文献改变。没有实施E0或新候选，没有启动训练或连接服务器。本轮只更新本文。

采用Exa Search，三个方向：中心—形状耦合与分项设计、几何代理与高精度误差响应、共享梯度取舍。共10次成功搜索，请求并返回60条候选，按URL去重46条；同一论文的不同版本/页面仍有重复，不能称60篇已读论文。重点核读KLD、KFIoU、ProbIoU、ProbIoU+、角度边界、各向异性高斯和PCGrad的方法/限制，并核对作者ProbIoU+代码及本地KFLoss/高斯损失；另核读GaussianDet的GEIoU分项，不将其其他模块引入本项目。这不是穷尽新颖性检索。

复用第17节完整TRAIN/VAL JSON与第15节源码。没有增加TEST阅读、模型推理、反传或完整审计。论文的测试集实验仅是外部文献证据，与本项目TEST用途分开。

### 18.2 文件支持的筛选事实

- 主要矛盾仍是D修复real严重错位/中心，但普通real尺寸、方向和sim角度未联合改善；不是已经证明的代码缺陷或普遍补偿—KLD反向梯度。
- 对原JSON的额外只读统计：VAL GT长宽比real最小/中位/最大为2.0950/2.2984/2.3050，sim为2.8461/2.9131/3.0724，没有<=1.2的GT；B/D所有有输出预测也没有<=1.2的框（B最小real1.7405、sim2.3208；D最小real1.9465、sim2.7035）。所以近方形失去方向虽是高斯表示的一般局限，却没有证据解释这批VAL退化；不能外推所有TRAIN或未来样本。
- 已记录的GT长轴角real范围-23.4435至26.4833度、sim为-20.4401至45.2803度，距le90的±90度边界至少10度的排除条件下没有边界附近GT。只能说这批VAL没有明显GT长轴边界聚集，不能排除anchor delta、edge_swap等其他编码路径问题。
- 后期TRAIN探针每视图共16个正样本，B的S均值clean0.006839、half0.026263，最大分别0.024502/0.065309；D均值0.009600/0.026156，最大0.067776/0.126011。这些是16张重复探针的正样本，不是全TRAIN。提示评审近零/中小误差区的梯度形式比修复已未触发的raw上限更贴近当前证据。

### 18.3 关键论文、可借鉴内容及项目边界

| 论文/主来源 | 核读内容与可借鉴机制 | 对本项目的限制/处理 |
|---|---|---|
| [Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence，NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/98f13708210194c475687be6106a3b84-Paper.pdf)，[作者HTML](https://arxiv.org/html/2106.01883) | 第3节式8/9及参数梯度：区分中心和协方差项，角度敏感性随长宽比变化；将参数耦合视为可帮助高精度优化的自调制机制。 | 当前代码是两个方向KL之和，并非单向KL。不能把论文对单向KL非对称性的讨论用来解释当前SymKLD偏小，也不能把全部耦合当缺陷。保留联合几何作为基础目标。 |
| [The KFIoU Loss for Rotated Object Detection](https://arxiv.org/html/2201.12558) | 第4.1/4.2节：中心定位与仅依赖协方差的形状项分开；第5.2节/表1比较不同中心项，第4.1节强调与SkewIoU的趋势一致而非值完全相等。多数据/检测器及较高IoU指标有实验依据。 | 支持中心/形状分别处理与代理一致性检查，但未验证“保留全部SymKLD再加形状项”。标准KFIoU包括中心项，不能直接接入本地整个KFLoss当E。高斯代理仍不是真实矩形RIoU，不保证尺寸/纯角度指标同时改善。 |
| [Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection](https://arxiv.org/html/2106.06072)，[作者代码入口](https://github.com/ProbIOU/probiou-sample) | 第3.3节式7–10明确分解Bhattacharyya中心B1和形状B2，并指出两项可不同加权；Hellinger给出有界相似性。文中指出Hellinger大差异时梯度小，建议BD到Hellinger的训练阶段切换。第4.3节有RetinaNet/R3Det与DOTA/HRSC对照。 | 可借鉴同中心B2及不同误差区响应。实验并非全面胜过GWD，DOTA R3Det亦有下降，不能只摘收益。第4.4节承认近方形方向信息和细长框数值风险。E-H只取同中心形状项，是本项目适配，不是论文整套方案复现；不新增阶段调度。 |
| [ProbIoU+: Enhanced Probabilistic IoU Loss for Oriented Object Detection](https://link.springer.com/chapter/10.1007/978-3-032-31920-3_38)，[作者loss代码](https://github.com/Sakasy/ProbIoU-plus/blob/main/mmrotate/models/losses/probiou_plus_loss.py) | 出版页First Online为2026-08-03，ICPR2026，卷册引用年为2027，不能混写。公开摘要指出Hellinger大差异饱和，作者代码加入归一化中心距离，并detach分母；代码核对证实这是新增中心监督。 | 全文为订阅预览，本轮只核验摘要及作者实现，不冒称已读完整消融。针对大中心偏差的补偿与当前D取舍不直接匹配，不采用整项。作者当前实现还未在forward中使用传入的avg_factor做本项目式聚合；不能原样复制破坏正样本归一化。 |
| [Rethinking Boundary Discontinuity Problem for Oriented Object Detection，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Rethinking_Boundary_Discontinuity_Problem_for_Oriented_Object_Detection_CVPR_2024_paper.pdf)，[作者HTML](https://arxiv.org/html/2305.10061) | 第3节区分平滑loss和连续输出编码，第4节提出ACM角度双优化。提醒高斯loss不保证输出无边界问题。 | 当前没有边界附近GT聚集证据；ACM改变输出编码和解码，超过E单因素范围，列为以后有对应证据时再评估的路线。不能由论文标题断言当前sim误差根因。 |
| [Enhancing Rotated Object Detection via Anisotropic Gaussian Bounding Box and Bhattacharyya Distance](https://arxiv.org/html/2510.16445)，[发表元数据](https://arxiv.org/abs/2510.16445)（Neurocomputing2025，623:129432） | 第3.1节用新的近方形映射保留方向，第3.2节使用BD并重加权中心，目标包括修复普通高斯对方形方向丢失。 | 当前VAL不是近方形，也非单向KL，论文的两个动机不能直接套用。更改高斯映射及新增中心权重扩大范围，不作为下一项。文中理论表述也需逐式核验；BD原量不是满足三角不等式的度量，不能照抄“BD天然是metric”的主张。 |
| [Gradient Surgery for Multi-Task Learning，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)，[作者HTML](https://arxiv.org/html/2001.06782v4) | 第2.2节明确说梯度冲突本身并不必然有害，讨论冲突、范数失衡、高曲率同时出现的条件；PCGrad改变梯度更新。 | 本项目只有冻结小样本和单参数共享探针，缺少上述条件及干预收益证据；原回归与D合计未普遍反向。故不加入PCGrad、动态loss平衡或拆塔。 |

另检索到[GaussianDet，Symmetry2025](https://www.mdpi.com/2073-8994/17/4/594)第4.4节的GEIoU同样采用BD中心/形状分解；其FBBox/ODAA等整体方法超出范围，不能因名称相近就认定是本项目SymEOOD的代码来源或独立验证E。GWD/GauCho已核读内容复用第15.4节，不追加表示/头改造。

### 18.4 E0与文献形状候选的具体差别（数学推导，不是收益证据）

E0仍是`L_B + 0.25*reduce(f(S))`，S为第15节同中心对称KL迹项，f(x)=sqrt(1+x)-1（另有限幅）。它只改变既有形状监督权重，不能仅由文献说已匹配RIoU或解决尺寸偏差。

ProbIoU可借鉴的同中心形状量为：

```text
M = (Sigma_pred + Sigma_gt) / 2
T = 0.5 * log[det(M) / sqrt(det(Sigma_pred)*det(Sigma_gt))]
H_shape = sqrt(1 - exp(-T))
候选 E-H: L_B + lambda_H * weighted_reduce(H_shape, global_positive_count)
```

T即论文B2；只读取协方差，不使用中心。H_shape是同中心高斯的Hellinger距离，不是整框ProbIoU，不是矩形RIoU。继续使用当前OBB高斯映射、正样本、权重及归一化。GT detach、空样本接图零、数值处理和浮点误差下非负性仍需实现验证。两个框在正常区共同等比例缩放时T/H不变；中心的直接新增偏导为0，共享参数仍可能影响中心、分类、输出和深度。

对同角度、同中心且尺寸误差很小的算例，令u=log(wp/wg)、v=log(hp/hg)，可由两种公式直接推导：

```text
S = cosh(2u) + cosh(2v) - 2 ~= 2*(u^2+v^2)
f(S) ~= u^2+v^2
T = 0.5*[log(cosh(u)) + log(cosh(v))] ~= (u^2+v^2)/4
H_shape ~= 0.5*sqrt(u^2+v^2)
```

所以E-H在近零区呈不同于E0的误差信号，值得针对普通框精度评审，而不只是增加同一项权重。但理想H_shape在完全一致处不是光滑的；稳定化eps会改变极小误差区梯度，必须检查有限值与主回归梯度比例。大差异饱和、异常主导或高曲率均可能带来代价。这里不宣称Hellinger一定提高精度，也不将论文的完整方法收益外推到本适配。`lambda_H`尚未确定，不能把E0的0.25当成等梯度强度直接照搬。

另一个文献候选是只取KFIoU协方差乘积形状量。其2D原始上限为1/3，论文展示趋势时乘3；标准loss的函数形式不只一种。本地`kf_iou_loss.py`同时带中心SmoothL1，直接使用会改变实验身份。若以后选择该候选，需明确只保留协方差项、函数/归一化与零点，并单独定强度，不能冒称原样KFIoU复现。它仍是平滑高斯代理，当前没有证据优于E0/E-H，暂列备选。

### 18.5 推荐顺序与边界（推断、待验证）

**结论：** 文献支持继续研究中心以外的尺寸—方向联合监督，也提示“损失值下降不等于实际几何变好”，不足以证明D根因或确定最优公式。此次不机械冻结E0；以E0为已有参考，优先评审E-H的普通误差区响应，KFIoU形状项为备选。不同时叠加多个新loss、不增加新中心项、不改教师/分配/推理/编码，不启动多个正式训练实验。

**下一步必要小检查（尚未执行）：** 先比较少量解析几何情形中的形状指标、真实RIoU及梯度方向：等比例尺寸偏小/偏大、单边误差、纯角度误差、角度与尺寸共同偏差；范围参照既有TRAIN/VAL而不重新调模型。固定这些算例用途为度量/梯度性质核验，不能据其宣布泛化收益或挑VAL最优系数。修正第17.3节直接梯度探针后，必要时复用同批TRAIN做有限参数级检查；不重跑D全套审计。正式训练前依据数学性质和TRAIN强度冻结一个公式、系数与数值稳定化规则，并记录依据；若E-H无法满足有限/合理梯度要求，则保留E0或暂停，不自动扫系数。

通过后只安排一次同初始化、同seed、同预算的B/最终E对照。第15.3节VAL选权契约及覆盖、连续性、联合几何门槛保持不变；同时报输出中心命中、输出覆盖和全帧正确覆盖，分开报告普通real与严重错位帧，检查尺寸signed bias/p90和sim纯角度。不能把论文AP或本项目总体RIoU代替这些目标。单次微小收益不称稳定/显著。

所有设计变化必须发生在实验前，并将E0和最终E的身份区分，不能同名覆盖历史方案。当前TEST已多次暴露，本轮未使用TEST，不据其定公式/系数或重选权重；等比例变换、原图还原和深度约束维持，几何改进不等同独立深度精度已验证。后续如获实施授权，只在本地改代码并交付服务器指令，不连接服务器。

## 19. E候选的局部实现、必要检查和代码审查（2026-10-01）

### 19.1 授权、实现范围与最新状态

用户授权按第18.5节完成检查、代码审查及服务器指令。本轮在本地实现独立候选损失和有限TRAIN探针，修复第17.3节的直接梯度测量问题；没有连接服务器、启动训练、更新权重或重选epoch。B epoch24仍是有效方案。

新增形状模块只由探针显式导入；没有接入B的head、训练配置或损失包默认导入。原SymKLD、分配、分类、推理、等比例增强、原图坐标还原及深度链路均未修改。没有重新开展D完整审计。本节取代第18节“未实施/尚未执行”的状态描述，但不代表正式E训练设计已冻结。

| 文件 | 本轮用途 |
|---|---|
| `mmrotate/models/losses/covariance_shape_loss.py` | 独立E0/E-H协方差形状候选，GT detach、空样本、权重及全局正样本归一化 |
| `crane_project/tools/preflight_port_shape_e_v1.py` | 解析几何检查；冻结B实际主损失正样本上的TRAIN数值/梯度检查 |
| `crane_project/tools/diagnose_port_center_size_d_v1.py` | 修复主KLD直接梯度的图节点捕获，不改变D模型或原结果 |
| `tests/test_port_shape_e_preflight.py` | 候选数学/梯度/协议/有限流程的CPU测试 |
| `tests/test_port_center_size_d_diagnosis.py` | 补充直接KLD图依赖及非零梯度回归断言 |

### 19.2 检查实现和冻结边界（事实）

E0采用`0.25*f(S)`，`f(S)=min(sqrt(1+S)-1,10)`，S为同中心对称KL迹项。E-H探针采用单位权重，T使用第18.4节同中心Bhattacharyya协方差项，稳定化形式固定为：

```text
H_eps = sqrt(-expm1(-T) + eps) - sqrt(eps)
eps = 1e-6
```

这是零点平滑的本项目适配，不是论文理想Hellinger的逐值复现。沿用当前高斯映射及边长最小1像素、det最小1e-6、逆矩阵限幅等保护；新增形状项的内部运算采用float64，再返回预测张量dtype。浮点精度与保护激活时，E0不是原float32 SymKLD形状分项的逐位相同实现。内部双精度只影响候选探针，不更改B原损失。报告分别记录边长/行列式/逆矩阵保护、负舍入、raw及loss限幅、Hellinger大误差尾部。

TRAIN探针只载入已按原VAL契约选定的B epoch24，验证sweep角色、B配置SHA、checkpoint SHA、内嵌配置、epoch24及seed0；不读取VAL预测或重选权重。复用原有限TRAIN检查的4张图片：索引`0/1810`和`905/2184`，分别为real_seq01_00000、sim_seq08_00000、real_seq06_00006、sim_seq08_00374。本地真实数据加载确认TRAIN为real1810+sim748，两尺度视图顺序相同。

每对图片分别检查clean与0.5等比例缩小、warmup O2M与late O2O，共8个batch、16次图像视图，只有4张独立图片。关闭随机翻转；通过调度计数器重放分配阶段。冻结的epoch24特征并不等同真实训练初期特征，不能据warmup重放代表初始化时梯度。

捕获实际主KLD使用的decoded正样本节点，以完全相同的正样本、权重和归一化计算E0与单位E-H。检查完整输出回归卷积、完整输出分类卷积、完整可训练FPN、一个backbone参数的范数/余弦，以及decoded xywhθ的直接梯度。FPN本轮是完整范数；backbone仍仅一个参数，不能称完整backbone检查。参数级报告来自real+sim混合batch，不是按域拆分的FPN梯度。

禁止非有限损失/梯度、丢失非空正样本图依赖、新增直接xy梯度或输出分类卷积直接梯度；要求两候选回归卷积和FPN信号非零。GT detach另由单元测试核验。共享参数仍会间接改变中心、分类和覆盖，直接xy为0不保证这些指标不变。

额外计算实际B、B+E0、B+单位E-H的完整目标裁剪前后范数。单位E-H仅用于强度/裁剪压力检查，不是建议的训练系数。没有optimizer step；记录参数SHA前后相同，调度计数器可变。描述性强度参考为各batch完整FPN的`norm(E0)/norm(unit E-H)`中位数，并报告范围。该参考不能自动冻结lambda_H，仍须看分配阶段/尺度差异、保护、回归与共享参数比例。

### 19.3 直接梯度探针修复和RIoU必要缺口（事实）

第17.3节原探针把主KLD对另一次正样本切片求导，`allow_unused=True`返回None后显示成0。修复后捕获主KLD实际节点，并核验D另一次切片数值/GT/权重/归一化相同，再在主节点上重算D分项、断言D标量相同。对非空节点的缺失依赖直接报错；新增依赖计数和按域非零KLD断言。旧JSON中主KLD的decoded/domain直接0仍是失效测量，不能改写为新测量；既有参数级梯度不受此节点问题影响。本轮没有重跑D服务器探针。

几何小算例还暴露本地OpenCV 4.13.0的近共线/共享边交集问题：例如理论RIoU为0.95的单边缩小框，既有`compute_riou`可算出约0.20，可能制造错误的下降方向结论。新增独立float64凸多边形裁剪用于机制检查，并以包含框、相交/分离、旋转正方形、对称性和大坐标平移的解析结果验证。不修改历史离线评估器、选权指标或报告。

只为排除该缺口，复用原D诊断JSON已保存的B/D VAL输出，与双精度结果交叉核对1772对框；没有重新推理。源JSON SHA256为`4062ce140b77a5e80b4d208c7cb71ceb1a7bcc84d413fd9bab25a153a7c3c487`。

| 固定输出组 | 框对数 | 最大RIoU绝对差 | 差值>1e-3 | RIoU≥0.5判断变化 |
|---|---:|---:|---:|---:|
| B real | 374 | 1.4677e-5 | 0 | 0 |
| B sim | 512 | 3.8857e-6 | 0 | 0 |
| D real | 374 | 3.1918e-6 | 0 | 0 |
| D sim | 512 | 3.0969e-6 | 0 | 0 |

该核对支持第17节这些缓存输出上的D取舍结论不受此误差影响，不能外推全部oracle替换、其他OpenCV版本或TEST。交叉核对保存在`work_dirs/port_shape_e_v1_existing_val_riou_crosscheck_20261001.json`。

### 19.4 本地验证和审查结论（事实及限制）

30项CPU测试通过，覆盖两模式/dtype、宽高交换+角度周期、正常区等比例缩放、中心直接梯度为零、尺寸与非方形角度有效梯度、GT detach、空样本、权重归一化、double gradcheck、极端尺寸保护、解析公式、错误图节点回归、冻结身份及输出防覆盖。有限流程使用真实head，但backbone/图像/CUDA I/O是CPU fixture；不代表真实ResNet+CUDA路径已通过。最后修正D探针临时图引用清理后，受影响16项D测试再次通过。

```bash
/opt/anaconda3/envs/mmrot/bin/python -m pytest -q \
  tests/test_port_shape_e_preflight.py tests/test_port_center_size_d_diagnosis.py
```

解析几何包括长宽比1/2.3/3，完全一致、±5%等比例尺寸、单长/短边缩小、2/5度角差、联合误差、大误差及只变中心，每种比较E0/单位E-H，共60条。最终使用独立float64 RIoU后，微小归一化下降步没有loss上升>1e-12或RIoU下降>1e-10。方形纯角差和只变中心对协方差形状项无辨识力，符合表示性质。有限算例不能证明任意误差的RIoU方向一致，更不能证明检测覆盖、VAL几何或泛化收益。

最终当前源码报告为`work_dirs/port_shape_e_v1_geometry_cpu_reviewed_20261001.json`，状态`GEOMETRY_ONLY_RUNTIME_UNVERIFIED`。初版`...geometry_cpu_20261001.json`保留但其OpenCV算例RIoU已由本节替代；`...fix1...`和`...final...`为中间快照，以reviewed报告中的源码SHA为本次交付依据。

代码审查未发现当前局部实现的阻断问题。审查修复了直接图依赖测量和D探针新增临时引用清理；新脚本捕获hook采用finally恢复，输出文件拒绝覆盖，失败保留已有进度。AST语法及diff空白检查通过。当前不能验证真实CUDA显存/速度、实际B权重的保护和梯度分布；这些由下面服务器短检查补齐。没有把通过CPU检查写成已验证E收益。

### 19.5 交付与服务器指令（待验证）

上传包：`work_dirs/port_shape_e_preflight_v1_20261001.tar.gz`，只含19.1节的5个代码/测试文件，路径相对项目根目录；不含模型、数据或训练配置。现有服务器项目需保留B/D配置、旧preflight依赖、TRAIN数据、B权重和原B VAL sweep。将包放在服务器项目根目录，解压后运行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
tar -xzf port_shape_e_preflight_v1_20261001.tar.gz
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_shape_e_v1.py \
  --gpu 0 \
  --out-json work_dirs/port_shape_e_v1_train_preflight.json
```

默认冻结身份目录为`work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1`；仅在原sweep确实移动时用`--reference-sweep`指定它，不得临时换epoch。物理卡3被映射为进程内gpu0。这个命令仅做8个batch检查，optimizer steps/训练epoch均为0，无VAL/TEST推理。

成功状态应为`TRAIN_PROBE_COMPLETE_REVIEW_REQUIRED`，正式配置字段仍为`NOT_FROZEN`。返回终端输出及`work_dirs/port_shape_e_v1_train_preflight.json`；若失败，另返回同名`.artifacts.json`和`.progress.jsonl`（如果已经生成）。脚本拒绝覆盖旧文件；重跑使用新的`--out-json`，无需删除旧证据。

**下一步判断规则：** 先看有限/连接/保护/裁剪，再看尺度和分配阶段的强度范围与回归/FPN方向。负余弦本身不等于有害。若E-H信号合理，再基于数学性质和TRAIN参考冻结一个公式及lambda_H，并登记后只进行一次同初始化、同seed、同预算的B/E对照；不自动照搬0.25，不扫描多个系数，也不先训练再按VAL改设计。E-H不满足要求时，记录原因后评估保留E0或暂停，不能自动改成另一套算法。

第15.3节VAL选权及覆盖、连续性、联合几何条件仍有效：中心命中率只统计输出帧，另报输出覆盖率及全帧中心正确覆盖；区分普通real与严重错位组，报尺寸signed bias/p90和sim纯角度。TEST已经多次暴露，本轮不用于公式、系数、门槛或权重调整。保持等比例、原图还原及深度估计约束，不把几何检查当独立深度准确性证据。

## 20. 服务器E候选TRAIN短检查结果与测试代码复核（2026-10-01）

### 20.1 结果身份和复核范围（事实）

用户返回终端输出及`/Users/mac/Downloads/port_shape_e_v1_train_preflight.json`，要求读取、检查测试代码、分析并建议下一步。本轮只复核结果和源码，保存证据及更新本文，没有改模型/探针/训练配置，没有连接服务器或启动训练。返回JSON已原字节保存为`work_dirs/port_shape_e_v1_train_preflight_server_20261001.json`，SHA256为`625a0dda8682b01c54337e98a798c31ae282ee15c73ebc5a00df09c84a387e7a`。

终端8行的32个loss数值与JSON逐项完全一致；报告中7个源码SHA均与当前本地文件匹配，4张TRAIN图片及两组TRAIN标注集合SHA也与本地一致。B冻结身份epoch24、iter15360、seed0，内嵌配置契约MATCH；checkpoint SHA为`8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23`。本地没有该权重，身份验证来自服务器脚本的SHA/配置检查，不能声称本地再次读取权重。

状态`TRAIN_PROBE_COMPLETE_REVIEW_REQUIRED`，正式配置仍`NOT_FROZEN`。服务器为GTX1080、PyTorch1.13.1+cu117，峰值allocated显存2489.919MiB；这不是reserved显存或正式训练速度/显存预测。参数SHA前后相同，optimizer steps=0、training epochs=0。

### 20.2 有效梯度、保护及裁剪（事实）

实际8个batch、16次图片视图，仍只有4张独立TRAIN图片。每个O2M batch18个正样本，O2O为2个，正常反映topk9到topk1，不是检测输出数量或输出覆盖率。80个正样本出现次数包含重复视图/分配，不是80张独立图片。

8个batch的实际主KLD非空节点连接计数均完整，两形状项w/h/角度梯度及回归卷积/FPN信号均非零；新增xy直接梯度和输出分类卷积直接梯度均为0。分类仍有共享特征和预测相关分配的间接变化可能，不能据此保证覆盖不变。全部JSON浮点数有限。

记录的边长下限、协方差/平均协方差det下限、逆矩阵上限、S/T负舍入及raw/loss上限、Hellinger尾部计数均为0；原KLD报告的det/inverse/raw/loss保护也为0。完整目标clip10均未触发：实际B裁剪前范数范围1.292919–7.399534，B+E0为1.425511–7.626419，B+单位E-H为3.899160–8.407352。裁剪前后微小差别属于计算精度，不能读成梯度更新。有限/未裁剪并不等于共享更新幅度或几何结果已合理。

### 20.3 分阶段强度：总体匹配中位数不能直接用作系数（事实及解释）

下表为完整FPN范数比，相对于当前batch主SymKLD，不是相对于含主分类/ATSS辅助项的完整B目标：

| 分配重放/尺度 | E0 / 主KLD | 单位E-H / 主KLD | 匹配E0的lambda_H范围 |
|---|---:|---:|---:|
| O2M / clean | 0.1164–0.1200 | 0.4284–0.5695 | 0.2106–0.2717 |
| O2M / half | 0.0907–0.1192 | 0.3383–0.3915 | 0.2681–0.3045 |
| O2O / clean | 0.0738–0.1194 | 2.2180–2.2240 | 0.0333–0.0537 |
| O2O / half | 0.0698–0.1210 | 0.8508–0.9867 | 0.0707–0.1422 |

脚本8个batch匹配系数的中位数0.1764368、范围0.0332759–0.3044739均独立复算一致，没有计算错误。但O2M中位数0.2699005、O2O中位数0.0622121；混合中位数由两个分配条件各占4个batch人为决定，不能当所有训练阶段等强度的证据。若直接采用0.1764，本次O2O的新增FPN范数约为主KLD的15.0%–39.2%；直接照搬0.25则为21.3%–55.6%。这些只是线性缩放的诊断描述，不是运行了这些系数的训练或完整裁剪检查。

**数学解释：** 第18.4节近零展开中E0约为0.25*r²，而平滑E-H约为`sqrt(r²/4+eps)-sqrt(eps)`，r²=u²+v²。因此两者单位权重的径向梯度比近似为`1/sqrt(r²+4*eps)`：在平滑区外，形状误差越小，E-H相对E0越强；完全一致处两者梯度仍为0。O2O的S均值clean约0.00239/0.00557、half约0.00974/0.03150，明显小于这些O2M正样本的0.10–0.38，与上述响应相容。FPN比例还受分配、中心项和共享映射影响，不能由这个展开精确预测或证明泛化收益。

完整回归卷积的E0/主KLD范围0.05098–0.11305（O2O），单位E-H为0.71069–2.02978（O2O），同样显示小误差区增强。FPN与主KLD的余弦：E0为0.5604–0.9644，E-H为0.5507–0.9438，8个batch均正；主回归卷积也均正。只能排除本探针范围内这两组参数的明显反向合成，不能保证real中心/分类/覆盖或普通几何不退化。

### 20.4 测试代码审查结论和限制

**审查结论（事实）：** 没有发现足以使这次结果失效的损失分解、权重/归一化、实际节点捕获、梯度范数、系数汇总或完整目标累加错误。逐项重算了范数比、匹配系数、中位数/范围及裁剪前后关系。捕获的是实际KLD正样本，修复后的第17.3节错误没有再次出现；每层normalizer相同，本次也等于实际总正样本数。完整目标累加包含`aux0_loss_*`，没有漏掉既有ATSS辅助项。

**需要保留的测量边界：**

1. `warmup_o2m`只是epoch24冻结参数上的调度/分配重放，不是同seed0初始化下的真实早期特征；探针seed1701用于可复现检查，checkpoint的seed0身份另已核验。因此真实初始化接入路径仍是正式实现时必要的小缺口。
2. `coefficient_reference.proposed_lambda`字段虽然有NOT_FROZEN说明，名字容易被误读；其含义始终是混合条件下的描述性匹配参考，不能自动用于训练。下一版汇总宜显式分阶段/尺度标注，保留本次原JSON。
3. backbone只测layer4末块一个参数；在两行O2O中主KLD/E0/E-H均为0，另一个half O2O行两候选余弦均约-0.3189。这不能代表完整backbone梯度缺失或E-H新增的普遍冲突。当前参数报告不区分None依赖和连接但数值为0；实际正样本节点另有显式依赖检查。无需据此追加全backbone审计。
4. 混合real/sim的FPN范数不能外推逐域；直接w/h的单位是像素、角度是弧度，不能比较列的绝对大小来断言“角度主导”。形状保护计数来自解码后框，不能据全部0排除bbox_coder的delta限幅或head解码上界等更早的保护。
5. 几何小算例是独立形状项的一步无量纲方向检查，没有测实际共享网络更新后的中心、RIoU、输出覆盖、连续性或深度精度。当前脚本通过状态是运行完成待评审，不是几何收益门槛通过。

本轮源码与已通过的第19节版本相同，复用该30项CPU测试，不无理由重复同版测试或D完整诊断。返回结果补齐真实B checkpoint/ResNet/CUDA路径在这8个batch的有限执行证据，不等同覆盖正式训练全过程。

### 20.5 下一步单一候选建议（推断、待验证；尚未接入或冻结正式配置）

推荐继续E-H，下一版采用固定`lambda_H=0.05`作为单一保守候选，保留本次H_eps、eps1e-6和局部float64，不新增中心补偿、阶段调度、动态平衡或其他形状项。选择依据是普通小误差区与后期主监督相对强度，而非VAL/TEST指标或寻找数值最优。

按现有向量线性缩放，0.05在本次O2O中的新增FPN范数约为主KLD的4.254%–11.120%，回归卷积为3.553%–10.149%；O2M则为FPN1.691%–2.847%、回归卷积1.339%–2.234%。因此它并非全程匹配E0，而是有意保持较弱的大误差/多正样本增强、在小误差区增加信号。这是机制和强度方面的实验理由，不证明0.05最佳、早期信号足够、覆盖一定安全或D根因已找到。

正式接入后只补一个有限检查：用B原初始化流程（预训练backbone+同seed0初始化检测头），复用相同4张TRAIN图片的clean/half O2M视图，无optimizer步骤；核对新增loss确实只在同一主正样本上聚合、B目标不变、有限梯度及完整裁剪，并与独立计算数值一致。结合初始化大误差情况，记录必要的decoder/上界保护，避免把本次解码后保护0误当完整证明。不重新跑当前epoch24探针，不扫系数，不补完整TRAIN审计。

若接入/初始化检查通过，预登记固定公式、系数、seed、预算和第15.3节VAL选权/覆盖/联合几何门槛，再进行一次正式B/E-H对照；保持同初始化/同预算，不把本次冻结B诊断写成已经训练了E。若初始化信号或数值有问题，先记录具体问题，不能事后按VAL调系数或自动加入阶段调度。

后续中心命中率只统计输出帧，另报输出覆盖与全帧中心正确覆盖，并保留连续性、普通real尺寸/方向及sim角度条件。TEST已多次暴露，本次未使用TEST，不据其改设计/系数/权重。等比例变换、原图坐标还原和深度估计约束保持；本次没有独立深度GT验证。

## 21. 固定E-H v1实现、代码审查及服务器运行（2026-10-01）

### 21.1 授权与冻结实验身份

用户授权修改实验E、代码检查并给服务器指令。本轮将第20.5节单一候选接入为独立E-H v1：

```text
T = 0.5 * [log det((Sigma_pred+Sigma_gt)/2)
           - 0.5*(log det Sigma_pred + log det Sigma_gt)]
H_eps = sqrt(-expm1(-T)+1e-6) - sqrt(1e-6)
L_E-H-v1 = L_B + 0.05 * weighted_reduce(H_eps, global_positive_count)
```

公式、系数0.05、eps1e-6、沿用当前高斯映射及候选内部float64已冻结。正式配置`crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py`、目录`work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1`。E0是旧参考候选，未训练；E-H v1不是E0或整框ProbIoU复现。0.05的依据仍是第20节TRAIN强度与数学性质，不是经VAL/TEST得到的最优值或D根因结论。

B全部原目标、辅助头、数据含seq06、等比例增强、seed0、ImageNet初始化、24epoch、SGD/lr/裁剪、分配、推理阈值、原图还原及深度链路保持。解析配置仅新增形状项及work_dir。正式E从原初始化训练，不从B/D权重续训。本地没有连接服务器、启动训练或新增VAL/TEST推理。

### 21.2 实现及代码审查（事实）

| 文件 | 内容 |
|---|---|
| `mmrotate/models/dense_heads/sym_eood_head.py` | 可选`shape_compensation`，复用主KLD实际decoded正样本张量、GT、权重和分母；输出`loss_shape_compensation`及非优化正样本计数 |
| `mmrotate/models/losses/__init__.py` | 注册现有参数为空的`CovarianceShapeLoss` |
| `mmrotate/models/losses/covariance_shape_loss.py` | 复用第19/20节已检查的H_eps实现，没有改公式/数值处理 |
| `crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py` | 继承B，显式固定Hellinger模式、0.05、eps及独立目录 |
| `crane_project/tools/preflight_port_shape_e_h_v1.py` | 原初始化路径下实际接入的4批TRAIN检查，不加载B快照 |
| `crane_project/tools/compare_port_shape_e_h_val_v1.py` | 固定所选B/E VAL缓存的配对报告、原门槛、指标数值交叉核对；不推理或选权 |
| `tests/test_port_shape_e_h_integration.py` | 配置、参数身份、真实head/推理/梯度、初始化流程CPU fixture、VAL分母及缓存流程 |

新增形状损失没有可训练参数或buffer，B/E state_dict键一致；显式禁止同一head叠加D中心补偿和E形状补偿，以保持实验身份。未启用形状项的B/D行为维持；没有改变loss默认模式用于本次E，配置明确传入hellinger和0.05。空正样本项接图零，GT在协方差内部detach；新增项不直接监督xy、只在原正样本归一化下聚合。

初始化探针按`tools/train.py`的seed0→build_detector→init_weights流程各构造当前B/E，要求参数SHA、state键及初始buffer值一致，再用E检测器进行4个batch（两对既有TRAIN图×clean/half），仅实际初始化O2M，不重跑epoch24探针。关闭形状项后在相同训练前向模式重算B原目标，核对主分类/回归和所有辅助loss不变；同批独立计算E项并要求复用实际主KLD节点。head临时hook与形状项开关均在finally恢复。

记录新增xywhθ直接梯度、完整主回归/分类卷积与完整FPN的范数及相对主KLD/完整B比例、余弦，记录完整B/E两目标裁剪前后范数。检测非有限/缺失依赖、非零中心或输出分类卷积梯度、缺少回归/FPN信号、参数更新等错误；不通过时停止并保留sidecar。正常完成仅表示实现检查待评审，不自动判断覆盖安全或正式训练收益。记录decoder delta宽高限幅、解码中心/边长上界及下界与协方差保护，补第20.4节的必要缺口；不增加全TRAIN或完整backbone审计。

VAL工具严格沿用原16/18/20/22/24候选、选权配置/metric协议及中心阈值15；B固定epoch24，E只读取已经冻结的选中权重。记录selection_info，包含fallback时也披露。配对报告保留输出中心命中/输出覆盖/全帧正确覆盖、普通real与历史严重错位、尺寸signed log bias/p90、纯角度、原序列/编号缺口下的连续性。新增门槛字段只表达第15.3节既定条件，不自动替换B；sim角度同时对固定2.1072及B的未舍入实际RMSE作严格下降比较，避免舍入造成虚假微小改善。单次seed0没有显著/稳定收益结论。

为防第19节发现的近共线RIoU误差，另对固定B/E输出做独立双精度交叉核对，仅报差值及0.5阈值变化。差值>1e-3或阈值变化时标记需要评审，原选权及主指标不改、不用另一个IoU重选epoch。

**审查结论：** 没有发现当前实现阻断问题。审查改为B/E相同grad-enabled训练前向、立即释放B比较图，减少后端路径差异的误报风险；核对正式loss只累加一次、统计计数不是优化项、原推理无新模块使用、D tuple索引兼容、配置继承和服务器CLI。没有把CPU fixture写成真实ImageNet/CUDA已通过。

### 21.3 本地验证与未执行部分

初轮48项测试通过（6项新集成+42项相关既有回归）；补VAL两项并完成审查修正后，8项集成测试通过，42项未受后续修改影响的既有测试复用，共50项当前检查覆盖。包括真实head O2M/O2O、同权重推理完全一致、原B损失逐层一致、新项实际节点/直接梯度、空正样本、D/E混用拒绝、真实MMDetection ConcatDataset包装下的4批初始化CPU流程fixture、输出防覆盖、VAL独立分母、缓存比较和fallback披露。

```bash
/opt/anaconda3/envs/mmrot/bin/python -m pytest -q \
  tests/test_port_shape_e_h_integration.py \
  tests/test_port_shape_e_preflight.py \
  tests/test_port_center_size_compensation.py \
  tests/test_port_center_size_d_diagnosis.py
```

配置检查、Python AST及diff空白检查通过。最终配置/源码快照为`work_dirs/port_shape_e_h_v1_config_final_20261001.json`，状态`CONFIG_ONLY_INITIALIZATION_UNVERIFIED`。真实ResNet ImageNet加载、CUDA强度/显存和初始化保护待服务器检查；CPU流程fixture替换了backbone/FPN/CUDA及图像I/O。当前B/E初始化一致的检查也不能证明恢复了未存档的历史B初始张量，正式比较沿用原配置/来源和seed协议，不冒称历史逐位复现。

### 21.4 上传及服务器顺序

上传包：`work_dirs/port_shape_e_h_v1_code_20261001.tar.gz`，相对项目根目录打包E配置、head/注册/损失、检查及VAL工具、所复用工具和4个相关测试；不含数据、权重或历史结果。旧B/D配置及结果保持原身份，本文第19/20节历史源码SHA不能改写成新head的SHA。

将包放到服务器项目根目录，先运行原初始化的有限检查：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
tar -xzf port_shape_e_h_v1_code_20261001.tar.gz
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_shape_e_h_v1.py \
  --gpu 0 \
  --out-json work_dirs/port_shape_e_h_v1_init_preflight.json
```

此命令4个E batch及对应B原目标重算，不做optimizer步骤；首次初始化依赖原ImageNet缓存/加载机制。成功状态`INITIALIZATION_CHECK_COMPLETE_REVIEW_REQUIRED`。返回终端及主JSON，失败时连同已有`.artifacts.json`/`.progress.jsonl`；重跑换新的输出名，不覆盖旧证据。

**短检查评审通过后**再执行下面正式训练，2张GPU与B相同，每卡batch2、seed0、24epoch；脚本`dist_train.sh`内已固定`--seed 0`。不传load_from、resume或改变lr/epoch。新实验目录应没有既有训练产物，已有产物时保留并先核对身份，不能直接重新启动覆盖。

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py 2
```

训练完成后，沿用既定5个epoch的原VAL选权；没有TEST参数：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py \
  --work-dir work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1 \
  --sweep-dir work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/val_sweep_port_v1 \
  --epochs 16 18 20 22 24 --center-thresh 15 --mcml-limit 5 --gpu 3
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/compare_port_shape_e_h_val_v1.py \
  --out-json work_dirs/port_shape_e_h_v1_val_compare.json
```

VAL返回sweep_results.json及比较JSON，以固定选权和预登记门槛决定是否联合改善；不因比较报告另选epoch，不在结果后放宽门槛或扫系数。若数值/接入检查失败先修复具体问题；未通过覆盖与联合几何条件保留B。当前TEST已经多次暴露，本轮没有连接或使用TEST；TEST不能调参/重选权重。保持等比例、原图还原及深度约束，E-H的收益与独立深度准确性仍待验证。

## 22. E-H真实初始化服务器检查评审与训练判断（2026-10-01）

### 22.1 已提供证据与复核范围（事实）

用户提供4行初始化终端结果、末尾`INITIALIZATION_CHECK_COMPLETE_REVIEW_REQUIRED`状态及`/Users/mac/Downloads/port_shape_e_h_v1_init_preflight.progress.jsonl`，询问是否可训练。JSONL有4条完整有限数值行，原字节保存为`work_dirs/port_shape_e_h_v1_init_preflight_server_20261001.progress.jsonl`，SHA256为`a18faa98a53df2017e682daa3167c8360e0bb88250ba7a3ef50b5af7be78908a`。本轮未连接服务器、未执行训练、未修改模型/系数/优化器。

4行图像身份、尺度、loss_shape、FPN比例、裁剪范数及倍率与终端一致；每行18个正样本、5层全局分母18，实际主节点复用及B原损失一致字段均true。新增xy直接梯度及输出分类卷积梯度均0，wh/角度/回归/FPN信号非零。记录的decoder delta/中心/边长保护及形状协方差保护均0。范数比及clip10倍率独立复算一致。

本次未提供主`port_shape_e_h_v1_init_preflight.json`或artifact，本地Downloads也未找到，故不能独立核验本次服务器源码SHA、最终参数SHA及完整初始化元数据。依据第21节已检查脚本和用户终端正常完成状态，这些检查在该脚本完成前应已执行通过；该推断不能写成“本地已读取完整元数据”。应保留主JSON作为本次正式实验前的身份归档，不要求重跑探针来补文件。

### 22.2 严重裁剪及E新增信号（事实）

| 尺度/图片对 | 主分类loss | 主KLD loss | E-H loss | 完整B裁剪前范数 | clip倍率 |
|---|---:|---:|---:|---:|---:|
| clean / seq01+sim00000 | 9467.136 | 1.0058 | 0.01277 | 1212340.375 | 8.2485e-6 |
| clean / seq06+sim00374 | 3566.903 | 0.9466 | 0.01371 | 396018.594 | 2.5251e-5 |
| half / seq01+sim00000 | 30230.578 | 2.5953 | 0.02155 | 4001913.250 | 2.4988e-6 |
| half / seq06+sim00374 | 28751.330 | 2.3585 | 0.01762 | 3786760.750 | 2.6408e-6 |

所有完整B/E目标均严重触发裁剪，到约10；B/E裁剪前范数在记录精度内相同。相同标量不等于两更新向量逐位相同，也不等于E完全没有梯度。

主分类输出卷积梯度范数为2.2347e5–2.0357e6，主回归卷积原KLD仅70.33–192.43；分类输出卷积自身已贡献完整B范数平方的约25.9%–31.8%。这支持当前初始化存在原分类相关大梯度的判断；没有逐项拆开主/辅助分类和共享层梯度，不能宣称完整巨大梯度的精确归因。关闭E的B原目标已有同等严重裁剪，不能说是E-H新增项造成或是已证明的历史B训练故障。

新增FPN/主KLD范数比为0.003927–0.009997（约0.393%–1.000%），回归卷积为0.003884–0.010188（约0.388%–1.019%）；FPN与主KLD余弦0.4303–0.8936，均正。新增FPN/完整B则仅2.31e-7–3.39e-6；在同一全局裁剪倍率下新增FPN分量范数约6.57e-7–1.07e-5，初始总更新中的几何信号很弱。这是有限初始化的观测，不是整个训练阶段的有效监督强度。全局裁剪同时压低原回归和E项，不能只针对E增大系数抵消；与主KLD相对比例也不会单靠统一裁剪改变。

### 22.3 判断与执行边界（推断、待验证）

**可以开始预登记的一次正式E-H v1对照。** 依据是接入/有限值/直接梯度检查通过，新增项没有造成目前的大梯度或保护激活，且同初始化B原目标已具有相同现象。这个结论是允许进行受控实验，不代表初始化状态理想、整个训练稳定或几何收益已验证。当前4批无optimizer步骤，不能证明大分类梯度会多久下降；第20节冻结B后期的较小loss/梯度也不能补出当前初始化的训练轨迹。

保持0.05、eps、24epoch、seed0、两卡/每卡batch2、SGD/lr及clip10，不改分类权重、padding或分配，不从B续训，不扫描E系数。若以后单独研究原初始化/分类的大梯度，必须以另一个B/E共同协议处理，不能在本次E单因素对照中只改E。

用既有每50 iter日志观察首个epoch的`loss_cls`、辅助分类、`loss_bbox`、`loss_shape_compensation`和`grad_norm`（MMCV OptimizerHook在grad_clip启用时记录的是裁剪前范数；本地源码已核对）。初始大值本身不当成E失败，也不当成“正常且无需关注”。若出现NaN/Inf或训练步骤报错，停止定位；若分类loss与极端裁剪持续不下降，优先检查初始化/分类链路，不因这些日志临时改E系数、lr或门槛。没有事后添加数值阈值来按日志挑“最好”的实验。

正式命令沿用第21.4节：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
bash tools/dist_train.sh \
  crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py 2
```

训练目录须是该新实验，保留已有产物，不重复启动覆盖。后续只用既定VAL候选16/18/20/22/24及原选择规则；中心命中只统计输出帧，另报输出覆盖和全帧正确覆盖，保持连续性及联合几何门槛。当前TEST已多次暴露，不参与此次训练设计/调参/重选权重；深度精度仍无独立GT验证。

## 23. E-H固定设计说明与后续VAL/TEST命令（2026-10-01）

用户要求说明固定设计并给出VAL/TEST服务器指令。本轮只核对现有配置、损失和评估脚本，补充记录；没有改模型代码、连接服务器或运行推理。

**事实：** 第21.1节公式与系数不变。协方差沿用B的`R(theta) diag((w/2)^2,(h/2)^2) R(theta)^T`映射；额外项只使用协方差，不使用均值中心。Hellinger模式、权重0.05、eps1e-6及局部float64固定，复用主KLD正样本、权重与全局正样本分母。保留B全部原目标与等比例增强等训练/推理设置，不叠加D，不从B权重续训。

**机制推断与待验证：** 额外项共同约束尺寸和方向，协方差表示对宽高交换及相应角度变换等价，近方形框方向敏感性较弱。它检验相对形状监督是否能改善几何精度，不证明D退化根因或本方案已有收益。直接中心梯度为零不保证共享网络中心/分类/覆盖不变；深度精度仍无独立验证。

以下命令在同一个服务器shell按阶段执行，使用物理GPU3；`ckpt_sweep.py`内部设置CUDA_VISIBLE_DEVICES，因此无需另加GPU映射。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
EH_CONFIG=crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py
EH_WORK=work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1
EH_SWEEP="$EH_WORK/val_sweep_port_v1"

# 阶段1：训练完成后，仅VAL推理与原规则选权。
python crane_project/tools/ckpt_sweep.py \
  --config "$EH_CONFIG" --work-dir "$EH_WORK" --sweep-dir "$EH_SWEEP" \
  --epochs 16 18 20 22 24 --center-thresh 15 --mcml-limit 5 --gpu 3

# 阶段2：读取固定B ep24与E已选VAL缓存，检查预登记条件，不重选权。
python crane_project/tools/compare_port_shape_e_h_val_v1.py \
  --e-sweep "$EH_SWEEP" \
  --out-json work_dirs/port_shape_e_h_v1_val_compare.json

# 阶段3：先评审VAL、冻结实验身份，再运行一次所选权重的TEST。
python crane_project/tools/ckpt_sweep.py \
  --config "$EH_CONFIG" --work-dir "$EH_WORK" --sweep-dir "$EH_SWEEP" \
  --final-test-from "$EH_SWEEP/sweep_results.json" \
  --center-thresh 15 --gpu 3

# 阶段4：只读该TEST缓存，区分输出中心命中/输出覆盖/全帧正确覆盖。
EH_EPOCH=$(basename "$(cat "$EH_SWEEP/selected_checkpoint.txt")" .pth)
python crane_project/tools/audit_port_test_subsets_v1.py \
  --gt-dir crane_project/data/crane_grab_port_day2night_v1/test/annfiles \
  --pred-dir "$EH_SWEEP/final_test/$EH_EPOCH/preds/Task1_grab" \
  --out-json work_dirs/port_shape_e_h_v1_test_subsets.json
```

阶段1、2完成后先阅读`sweep_results.json`的`selection_info`（包括fallback）和比较报告；覆盖、连续性及联合几何门槛不放宽，不根据比较另挑epoch。比较工具依赖服务器已有B原VAL缓存，默认B目录是`work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1`，已迁移则明确传`--b-sweep`原缓存目录。

阶段3仅从原VAL选权JSON读取权重并核对身份，不扫描TEST epoch。TEST报告位于`$EH_SWEEP/final_test/$EH_EPOCH/final_test_metrics_v2.json`，逐帧缓存同目录`preds/`。阶段4是已有逐序列统计工具，不是新完整审计/候选归因；分别报告real_seq03、real_seq04和sim_seq09。原终端`R_center`是全帧正确覆盖，不能称为输出帧中心命中率。分序列字段依次为`output_conditional_center_hit_pct`、`output_coverage_pct`、`all_gt_frame_center_hit_pct`。合并real时应按计数/分母加权，不直接平均两序列百分比。

保留已有报告；VAL扫描及比较工具拒绝覆盖已有结果，已完成则直接复用。TEST当前已多次暴露，只能报告冻结方案在该已暴露集合的表现，不据其改公式、系数、阈值或权重。未满足VAL目标仍保留B；如另报告E-H负结果，沿用相同冻结权重并披露身份。

## 24. E-H v1正式VAL结果与是否继续TEST（2026-10-01）

### 24.1 输入、核验及证据范围（事实）

用户提供VAL扫描/比较终端附件`/Users/mac/.codex/attachments/5c1a3b48-d1f1-4d94-81e0-7d33082b1eb8/已粘贴的文本.txt`及完整`/Users/mac/Downloads/port_shape_e_h_v1_val_compare.json`。原JSON SHA256：`474434c3d8ca9967939dfdecc0e0c16aabdc985bd8bd0bcdc13f8c43d6e568f4`，原字节保存为`work_dirs/port_shape_e_h_v1_val_compare_server_20261001.json`。

终端5个候选为16/18/20/22/24，4个可行，epoch22按原硬约束及软评分选中，没有fallback（`constraint_pass`）；B固定epoch24。比较JSON的选中权重身份与终端一致，两域GT标注SHA相同，B/E配置SHA与本地原配置相同。报告记录E ep22权重SHA `b61c5fed3fdec9b70b0c1ecf91699bff1a01e05ee665fb0eff593fb7e7d1ca5a`。本地没有服务器权重/PKL/sweep原文件，不能独立重算其SHA或完整5候选选权；依据已提供终端和报告核对身份。

JSON包含两臂各887条逐帧框（real375、sim512）。逐帧图像ID/图像SHA/GT一致，独立复算输出、15px中心命中、长短边相对误差及signed log ratio，并用标准库重算配对/普通real组的均值、中位数、p90、RMSE及RIoU聚合，均与报告一致。几何失败条件独立复算一致。双精度RIoU交叉核对最大差异B 1.4677e-5、E 3.3156e-6，均无>1e-3项或0.5阈值变更，`metric_consistency_review_required=false`。这些输出上没有支持“此次几何退化由已知IoU数值问题造成”的证据。

本轮未连接服务器、训练、推理、改代码、选其他epoch或读取新增TEST。主训练日志未提供，不能据当前VAL归因训练期梯度强度/裁剪或收敛故障。

### 24.2 冻结B ep24 / E-H ep22的结果（事实）

| 指标 | B ep24 | E-H ep22 |
|---|---:|---:|
| real输出覆盖 | 374/375，99.7333% | 375/375，100% |
| real输出帧中心命中 | 360/374，96.2567% | 370/375，98.6667% |
| real全帧中心正确覆盖 | 360/375，96.0000% | 370/375，98.6667% |
| real最长无输出/RIoU失败 | 1 / 4 | 0 / 1 |
| real全帧平均RIoU | 0.795505 | 0.788410 |
| real共同输出长/短边平均相对误差，n374 | 8.4208% / 8.7580% | 10.3289% / 9.5451% |
| real共同输出长/短边误差p90 | 16.5210% / 16.0180% | 20.5295% / 18.0957% |
| real共同输出中心mean / RMSE，px | 9.5746 / 34.5779 | 9.0253 / 60.9479 |
| sim三项覆盖 | 各512/512，100% | 各512/512，100% |
| sim纯角度RMSE | 2.107151° | 1.706917° |
| sim全帧平均RIoU | 0.885695 | 0.873045 |
| sim长/短边平均相对误差，n512 | 3.8712% / 4.1388% | 5.2978% / 5.1285% |
| sim长/短边误差p90 | 7.7625% / 7.5082% | 9.9034% / 9.5393% |
| sim中心mean / p90，px | 1.6452 / 2.9585 | 1.8336 / 3.3853 |

sim纯角度RMSE相对下降约18.99%；尺寸和RIoU退化不因sim中心仍全部命中而消失。两域长/短边的mean signed log ratio都更负：real长 -0.04755→-0.08196、短 -0.05762→-0.07488；sim长 -0.02267→-0.04972、短 -0.03175→-0.05042，支持整体尺度偏小偏差加重，但并非每帧都变小/退化。

预登记15项检查中7项失败：两域共同输出长/短边误差下降共4项、两域全帧RIoU不降共2项、sim共同输出中心不劣1项；`all_conditions_met=false`。覆盖、最长失败及sim角度条件通过，不代表联合几何目标通过。

### 24.3 严重错位修复与普通框/尾部代价（事实）

B历史10个严重错位帧中9个被修复（中心<15且RIoU>=0.5），`real_seq07_00026`仍失败。包含补回B无输出帧在内，E-H共修复14个中心失败，同时新增4个：`real_seq07_00004`、`00009`、`00150`、`00190`。real总中心正确数净增10。RIoU>=0.5方面修复10个、新增4个，正确帧364→370。

新增失败的中心误差分别约57.48、127.19、905.82、725.47px；其中00150由B 1.87px变905.82px，00190由B 7.34px变725.47px，两帧E的RIoU均0。它们是已报告输出框的实际尾部失败，不是无输出；仅凭框无法断言错检对象身份或分类/分配根因。

排除原先已固定的10个B严重错位帧，普通real共同输出n364的RIoU 0.819545→0.793337；长/短边平均相对误差7.8524%/7.9396%→10.0509%/9.2536%；角度RMSE 2.232661°→2.382857°；中心mean 4.4375→8.8132px、RMSE 5.2923→61.4414px。普通组中心中位数3.6827→3.3901px、p90 7.8297→7.5622px改善，但上述新大偏移使均值/RMSE变差。全real平均中心误差轻微下降不能写成定位尾部风险得到保持。

seq07全帧RIoU 0.737034→0.721660，长短边误差11.3629%/12.5475%→14.6164%/13.7206%；seq14 RIoU 0.884194→0.889655且中心改善，长边误差略降、短边误差略增。退化并非每个real序列一致，不用改善序列掩盖总体结果。GT尺寸替换的RIoU均值增益在共同输出real 0.07243→0.10177、sim 0.03020→0.04659，仅作当前框尺寸影响的描述，不能当训练根因的因果证据。

### 24.4 判断、TEST建议与有限后续（推断、待验证）

**E-H v1没有达到联合优化目标，保留B；当前不建议继续以最终候选确认目的运行TEST。** 这是原预登记条件下的判断，没有事后放宽条件。epoch22仍是本实验正式VAL所选权重，不改选18或24来规避尺寸退化，也不把当前结果改称单独角度优化成功。

**机制推断：** 在这次单seed/不同VAL所选epoch比较中，额外协方差项的结果表现为sim方向及real严重错位改善，伴随尺寸低估和部分定位尾部代价。它不证明系数太大/太小、形状监督无效、中心补偿是D失败根因或初始化大分类梯度导致本次退化。原选权软评分没有直接长短边误差/平均RIoU项，因此选权合格与联合几何条件失败可同时成立；不能事后为E改变选权协议。

下一步先复用本轮训练日志，有限核对后期主回归/形状loss、分类及裁剪轨迹；再在已有VAL框/原图上核对seq07尺寸低估与新增00150/00190大偏移。目的只补TRAIN/VAL机制缺口，不重做完整审计、不恢复候选排序、不立即扫系数或新训练。日志loss比本身不等于参数梯度比；若确需接收端梯度，再另行设计有限TRAIN检查。当前收益/取舍不能外推深度误差，也没有多seed显著/稳定性证据。

若论文需要完整负结果或冻结取舍报告，可以单独明确决定运行一次固定epoch22的TEST；这属于结果报告，不是继续寻找可替换B的证据。无论其结果好坏，都不据TEST调公式、系数、阈值、选其他epoch或推翻本次VAL门槛。当前TEST已多次暴露，必须披露，不能称为未接触的独立确认集。本轮没有启动该TEST。

## 25. 用户授权固定E-H epoch22执行TEST（2026-10-01）

用户指出视频数据在VAL/TEST不同序列上可能表现不同，明确要求仍进行一次TEST。该授权更新第24节“不建议继续”的执行建议：现在交付固定E-H ep22的服务器TEST及已有缓存逐序列统计命令，不再请求确认。第24节VAL未通过联合条件的事实保持，当前有效方案仍为B。不同序列可能存在差异是合理待验证问题，不预先假定TEST一定更好。

固定配置/系数/阈值，读取原VAL sweep所选epoch22，不扫描其他epoch，不根据TEST调整损失或改选权。原TEST已多次暴露，后续报告应披露。测试由用户在服务器执行，本地不连接服务器、不运行推理、不改模型或评估代码。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python crane_project/tools/ckpt_sweep.py \
  --config crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py \
  --work-dir work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1 \
  --sweep-dir work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/val_sweep_port_v1 \
  --final-test-from work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/val_sweep_port_v1/sweep_results.json \
  --center-thresh 15 --gpu 3

# 仅在上一步成功完成后执行：只读TEST TXT，不再推理。
python crane_project/tools/audit_port_test_subsets_v1.py \
  --gt-dir crane_project/data/crane_grab_port_day2night_v1/test/annfiles \
  --pred-dir work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/val_sweep_port_v1/final_test/epoch_22/preds/Task1_grab \
  --out-json work_dirs/port_shape_e_h_v1_test_subsets.json
```

物理GPU3由ckpt_sweep内部映射，无需额外设置CUDA_VISIBLE_DEVICES。保留原VAL sweep与全部TEST产物，不覆盖已有不同身份报告。主结果：`work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/val_sweep_port_v1/final_test/epoch_22/final_test_metrics_v2.json`；逐序列结果：`work_dirs/port_shape_e_h_v1_test_subsets.json`。带回两个JSON及终端打印；同时保留`preds/results.pkl`、`preds/Task1_grab/`供尺寸/定位逐帧分析。现有逐序列工具输出覆盖、RIoU、时序及原离线指标，不应冒称已新增尺寸误差分布报告。

后续以B既有固定TEST产物为对照，分别报告real_seq03、夜间real_seq04、sim_seq09；中心命中以输出帧为分母，同时报输出覆盖和全帧中心正确覆盖。连续性使用既有序列/编号缺口规则，最长RIoU失败不等同最长无输出。合并real使用帧数/命中计数加权。不能把视频相邻帧当独立重复实验来宣称显著性，也不能把几何结果直接写成深度准确性验证。冻结模型的TEST表现与第24节VAL取舍并列记录。

## 26. E-H epoch22冻结TEST结果与后续判断（2026-10-01）

### 26.1 已读证据与一致性检查（事实）

用户提供`/Users/mac/Downloads/final_test_metrics_v2.json`、`/Users/mac/Downloads/port_shape_e_h_v1_test_subsets.json`及终端附件`/Users/mac/.codex/attachments/664fa336-d299-4c0e-9265-3f65e4a8cf20/已粘贴的文本.txt`。

原字节归档：`work_dirs/port_shape_e_h_v1_final_test_metrics_server_20261001.json`，SHA256 `028fe55ea7e7533ef6d6b4e54196864dacf02afa48def578f2211096848805e4`；`work_dirs/port_shape_e_h_v1_test_subsets_server_20261001.json`，SHA256 `1ae978f08e992365ec93e5bc8af29b65ee69793119b6cee14a2ecccb4a63c902`。

主JSON为metric_protocol_version2、fixed_test_after_source_val_selection、epoch22、15px中心阈值、1440帧。配置SHA与本地固定E-H配置一致，权重SHA `b61c5fed3fdec9b70b0c1ecf91699bff1a01e05ee665fb0eff593fb7e7d1ca5a` 与第24节VAL报告相同。TEST标注SHA记录为`e0dbb1bd8aea7209314d8ed60bc44e965550ed606135cc0016e1075d717de13e`。本地无该TEST PKL/TXT与服务器权重，不能独立重算预测/权重SHA或逐帧几何，只核对报告身份及汇总一致性。

终端完整逐序列JSON与附件JSON逐字段相同。独立反算每序列整数输出、中心/IoU正确计数，复核比例、缺失输出的RIoU零值归一化、失败区间长度/总数/最大值、两real序列按帧加权后的中心及RIoU，与主JSON一致；输出1432=668+192+572，与终端转换框数一致。MMCV提示为版本迁移warning，终端未见推理失败。

B参考仍是第10节原B ep24固定TEST终端；本轮重新读取原附件`/Users/mac/.codex/attachments/3acfd6bd-c70c-4b04-9f21-8ea9bd67b0a3/已粘贴的文本.txt`核对1431框及原指标。未找到B新划分的逐序列JSON/完整预测，本轮不能作B/E逐帧交集或断言具体哪些B缺失被E修复。不能拿本地旧K1/旧划分TEST缓存冒充本次B。

本轮只分析已授权的固定TEST结果、现有VAL与源码定义，未连接服务器、修改代码、训练或重新推理，不据TEST选其他epoch或设计系数。

### 26.2 B / E-H固定TEST总体对照（事实）

| 指标 | B ep24 | E-H ep22 | 取舍 |
|---|---:|---:|---|
| 总输出帧 | 1431 | 1432 | 多1帧，但无配对修复证明 |
| real输出覆盖 | 859/868，98.9631% | 860/868，99.0783% | 略增 |
| real输出条件中心命中 | 858/859，99.8836% | 858/860，99.7674% | 略降 |
| real全帧中心正确覆盖 | 858/868，98.8479% | 858/868，98.8479% | 相同 |
| real全帧mean RIoU | 0.8169 | 0.8166 | 基本持平，报告值略降 |
| real DFR | 2.5103 | 2.5983 | 略退化 |
| real ACI | 0.9440 | 0.9427 | 略退化 |
| real MCML max / mean | 4 / 3.5 | 4 / 2.5 | 最长相同，均值改善 |
| real MRF | 2.29 | 2.00 | 改善 |
| sim三项覆盖 | 均100% | 均100% | 相同 |
| sim角度RMSE | 1.8823° | 2.0353° | 增0.1530°，约8.13% |
| sim全帧mean RIoU | 0.8816 | 0.8635 | 降0.0181 |
| sim DFR | 2.4734 | 2.6200 | 退化 |
| sim ACI | 0.9479 | 0.9502 | 改善 |
| sim MCML max / mean | 0 / 0 | 0 / 0 | 相同 |
| 两域TDR_w10 | 各100% | 各100% | 相同，不能视为全帧无失败 |

B real输出/中心数为原终端+固定分母反算；E为本次逐序列直接反算。E real有输出但中心不正确为2帧，B原汇总为1帧；不能据此认为“新增那一帧就是错误输出”，因缺少配对输出身份。real RIoU差值仅约0.0003，不宣称显著退化或改善；sim变化是本次序列上的观测，也没有多seed/独立序列显著性结论。

原sim A-RMSE为10px中心门控，未满足中心或无输出会计90°，不能通常等同纯角度。在本次sim572帧中，若哪怕1帧受90°惩罚，RMSE至少90/sqrt572=3.7631°；现有2.0353°（B 1.8823°也低于该界）排除了这种惩罚，因此依照当前完整分母实现，本次sim角度变化不能解释为门控/漏检惩罚增加。仍没有读取逐帧角度框直接重算。

### 26.3 E-H各视频序列覆盖与连续性（事实）

| 序列 | 输出/总帧 | 输出条件中心命中 | 全帧中心正确覆盖 | 全帧RIoU | 最长无输出 / RIoU失败 |
|---|---:|---:|---:|---:|---:|
| real_seq03 | 192/200，96% | 191/192，99.4792% | 191/200，95.5% | 0.748474 | 4 / 4 |
| real_seq04，夜间 | 668/668，100% | 667/668，99.8503% | 667/668，99.8503% | 0.837035 | 0 / 1 |
| sim_seq09 | 572/572，100% | 572/572，100% | 572/572，100% | 0.863504 | 0 / 0 |

8个无输出全部来自seq03：110、112、142–145、185、188。其RIoU失败为110–112（含有输出的111）、142–145、185、188，共9帧。seq03有输出条件RIoU 0.779660，全帧因8个缺失变0.748474，不能只报99.48%输出中心命中掩盖96%输出覆盖。

夜间seq04全部有输出，仅56帧RIoU失败；seq03 111帧为有输出但RIoU失败。两序列均各有1个输出框中心未达标，但报告没有列中心失败帧ID，不能仅因中心/IoU正确计数相等就认定中心失败也是111/56帧。未提供框及图像，不能断言只是微小中心偏差、错检对象或具体大小/方向原因。夜间seq04当前表现较好，但无B同序列数据，不能宣称它比B改善。

real MCML<=5通过不等于用户全帧正确覆盖目标完全实现，仍有4帧连续无输出。TDR_w10定义为每10帧窗口至少1帧RIoU>=0.5，所以100%与上述缺失并不冲突；DFR/ACI只统计连续且都有框的帧间，缺失会断开时序，不能单独作为无间断输出证明。MCML mean是既有分段最长失败的均值，不是所有失败区间长度的简单平均。

### 26.4 VAL/TEST解释及下一步（推断、待验证）

**保留B；E-H v1记录为未达到联合目标的固定实验，不作为最终方案。** VAL的sim角度2.107151→1.706917改善没有在TEST sim_seq09重复（1.8823→2.0353），两集sim RIoU均低于B。real在VAL覆盖改善，TEST全帧中心正确数保持而非提高，最长失败也未进一步缩短。不同视频序列确实表现不同，做这次冻结TEST有价值；它没有提供整体替换B的证据。不能据此证明过拟合、协方差角度耦合、权重过大/过小或训练梯度故障是根因。

TEST附件只有指标/区间，没有宽高误差、中心距离分布或原始框。因此无法断言TEST的RIoU下降具体由尺寸低估造成，也不能把第24节VAL尺寸结论直接移植到TEST。保持等比例/原图还原并未改变，独立深度准确性仍未验证。

下一步优先补TRAIN/VAL机制证据：复用本次已有训练日志看后期分类、主KLD、新增H及裁剪轨迹；若仍不足，再在既定有限TRAIN clean/half样本和冻结B/E权重上核对主KLD与E-H对宽高、角度的梯度方向/强度，不更新参数、不增加全TRAIN审计。梯度检查目的为区分有限样本上尺寸纠偏响应、角度响应与主损失交互，不凭loss比或单个夹角定训练根因。根据TRAIN/VAL证据再预登记下一项有限实验，不直接扫E系数、加中心补偿或改选权协议。

只为完善冻结TEST结果，必要时可读已有B/E TXT并补尺寸/定位分布，不需要重跑推理；这一步不用于新设计/系数/阈值选择。B逐序列JSON或固定缓存缺失时可由服务器已有产物补齐以公平描述视频差异，当前无需为了保留B的判断追加新TEST运行。当前TEST已多次暴露，须披露；不能把相邻帧当独立重复或把此次比较写成未接触测试集上的显著/稳定收益。

## 27. 只读TRAIN日志复用工具、检查与服务器指令（2026-10-01）

### 27.1 授权、实现及证据边界（事实）

用户要求按第26节建议复用已有训练日志核对后期分类、主KLD、形状损失及裁剪轨迹，并给服务器指令。本轮新增`crane_project/tools/analyze_port_shape_e_h_train_logs_v1.py`与相关测试，使用Python标准库，不导入torch/mmcv，不加载权重/数据，不推理或执行optimizer，不改模型/配置/日志记录方式，不访问VAL/TEST指标，不连接服务器。

工具读取MMCV TextLoggerHook实际生成的逐行JSON `.log.json`，仅mode=train纳入分析。元数据按seed0、B/E实验名、主SymKLDLoss/权重2、形状公式/系数、24epoch、norm2/clip10及work_dir核对；读取的是日志中已解析配置，使用受限AST字面量解析，不执行其中代码。明确冲突报错；元数据缺失/不能解析保留为需要评审，不能冒称完成原训练身份复现。每个输入原日志SHA及config文本SHA写入报告。

只有一份日志时自动发现；同目录多份不选“最新”，要求显式指定，不能把不同实验/重训混成轨迹。同一run分段日志可显式列出，但重复epoch/iter或单文件倒退立即拒绝，不覆盖记录或静默去重。B日志不存在时仍分析E并明确B比较不可用；默认不重建B历史曲线。JSON损坏/无TRAIN行立即报错，不忽略坏尾行。

按每epoch及固定1、2–4、5–16、17–24、17–22、23–24窗口报告mean/median/p90/min/max与缺失数；保留全部日志行的来源和轨迹。字段包括主loss_cls、主loss_bbox（KLD）、实际loss_shape_compensation、其他原loss及辅助分类/回归合计、总loss、lr、grad_norm和正样本计数（若存在）。新增形状/主KLD比是已加权窗口标量之比，不再次乘0.05；不将它等同参数梯度比。空/缺失项不填零，NaN/Inf/非标量记null并保存事件位置/原表示，输出严格有限JSON。另保存零值计数、epoch覆盖及总loss与可用组件和之差，供人工判断归零/舍入/缺口。

**裁剪口径修正：** 本地核对MMCV OptimizerHook记录的是每步裁剪前范数；LoggerHook在每个日志窗口调用log_buffer.average，再由TextLoggerHook写入JSON且舍入。因此工具统计的是“日志窗口平均裁剪前范数>10的占比”，不是实际每步裁剪频率，也不计算10/平均范数作为平均裁剪倍率。窗口均值<=10不能证明该窗口没有裁剪。默认日志间隔50，日志轨迹无法补出逐step的短尖峰或精确裁剪轨迹，当前不为此重训。

现有主头对NaN/Inf主分类/KLD已有归零保护，有限日志不证明保护从未触发；零值也可能来自舍入/样本情况，不能自动判为隐藏错误。loss曲线与grad_norm只提供训练数值/阶段线索，不直接证明宽高或角度梯度来源、泛化稳定性或几何失败根因。读取旧K1日志仅用于验证真实MMCV格式/AST解析，没有把其旧划分指标纳入B/E分析。

### 27.2 验证与审查（事实）

新增`tests/test_port_shape_e_h_train_logs.py`：12项标准库CPU检查通过，并在本地Python3.8环境验证兼容。覆盖TRAIN/VAL隔离、已加权比值/辅助聚合、窗口范数与实际裁剪口径区别、缺失/NaN/Inf/零分母、错误身份/系数、配置代码不可执行、重复/倒退日志、分段排序/来源、歧义发现、epoch缺口、B缺失/描述性对比、坏JSON/无TRAIN及真实CLI输出/拒绝覆盖。Python AST、diff空白检查及上传包内容逐字节校验通过。

本地没有本次B/E正式训练日志，尚无新的训练轨迹结论；测试是流程/统计fixture，不是服务器数值结果。未改训练、增强、正样本、归一化、梯度裁剪、推理或几何/深度链路。本次只在工具内生成新分析报告，不修改输入日志。

上传包：`work_dirs/port_shape_e_h_v1_train_log_tools_20261001.tar.gz`，含上述工具与测试2个文件，无数据、权重或日志，SHA256 `276a1013103a455bcf1c9bd23d2defe14610bf6d8bbd269acae36910dfdcbc73`。工具SHA `7103ebf0bfcc68a3a4ca7e871ae4bfaac6eb44ac0152fa9bd1c35b0e7ed268c6`。

### 27.3 服务器只读命令

把压缩包上传项目根目录后：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
tar -xzf port_shape_e_h_v1_train_log_tools_20261001.tar.gz

python crane_project/tools/analyze_port_shape_e_h_train_logs_v1.py \
  --out-json work_dirs/port_shape_e_h_v1_train_log_review.json
```

默认搜索E目录`work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1`及B目录`work_dirs/crane_symeood_k1_port_day2night_aug_b_v1`的`*.log.json`。不需GPU/PYTHONPATH/模型依赖。若列出多份日志，则指定该正式run的真实时间戳文件（以下时间戳是占位示例）：

```bash
python crane_project/tools/analyze_port_shape_e_h_train_logs_v1.py \
  --e-log work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/YYYYMMDD_HHMMSS.log.json \
  --b-log work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/YYYYMMDD_HHMMSS.log.json \
  --out-json work_dirs/port_shape_e_h_v1_train_log_review.json
```

如果B日志确实无法取得或需要明确先分析E，使用`--no-b`，仍须在E多份时指定`--e-log`；报告注明无B轨迹，不因此新训练B。断点分段若确属同一run可在同一`--e-log`/`--b-log`后依次列多个文件，必须无重叠。已有输出不覆盖，先复用；新输入/修复后需重新出报告则使用新的输出名。

带回完整`work_dirs/port_shape_e_h_v1_train_log_review.json`和终端窗口汇总。评审重点为首次大分类/大梯度是否回落、17–22及23–24的主/辅助分类、KLD、形状及正样本数、LR阶段及窗口梯度是否存在持续异常。单纯loss变小不等于尺寸/定位变好，也不能据形状loss比直接增大系数。如果日志已解释早期数值而无法解释几何退化，再按第26.4节有限TRAIN梯度检查补缺口，当前不先扩展全审计或启动新实验。B继续为有效方案，TEST多次暴露边界保持。
