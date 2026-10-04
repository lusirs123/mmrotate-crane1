# SymEOOD 几何精度优化交接（2026-10-01）

> 用途：在新对话中继续几何精度优化。本文汇总本轮对话、现有代码及收到的实验结果；保留事实、推断和待验证内容的区别。几何优化后续优先更新本文，不再为每次改动创建交接文件；可靠性另在[独立可靠性交接](reliability_handoff_20261003.md)维护。
>
> **2026-10-04最新执行见第75节：已新增仅复用选中正式头VAL的短边/对角线静态与时序残差诊断入口，15项CPU检查通过，并已在收到的结果包中完成本地只读验证。没有新增训练、平滑或TEST分析。诊断区分GT变化与预测波动：real DFR升高但GT相对对角线变化残差略改善，sim短边/对角线变化残差增大；不能直接以压低DFR作为训练目标。**
>
> **2026-10-04最新核验见第74节：正式结果包已读取，原VAL规则选中head_epoch_23；24轮VAL逐帧汇总与独立在线VAL一致，TEST汇总/分母/纯角度已核对。sim定位/纯角度/RIoU改善，real定位改善，但尺寸多数退化、VAL real RIoU下降、两域DFR上升。第73节及此前“仅终端/待JSON核验”属于当时证据状态。B仍为冻结前端和配对参考，可靠性另一工作线不自动切换。**
>
> **最新执行授权以第70节为准（2026-10-03）：用户明确要求先开展正式训练。本轮已实现独立的midpoint-formal v1入口：复用并冻结SymEOOD＋B的VAL epoch24，从零修正初始化的新头出发，在全部TRAIN的1.0/.5视图上训练24轮，完整VAL按原规则选权。不是重新训练B或端到端联合微调，不加载短fit TEST头、不加入时序/C2/可靠性新机制。原第66节probe失败和第68节探索性TEST数值保持；新候选尚未在服务器训练，B仍是保留前端。下段“未放行正式训练”属于这次新授权之前的记录，不能据此否定第70节的新授权，也不能反过来把新授权写成已验证收益。**
>
> **2026-10-03最新任务分工：本对话继续检测几何优化，可靠性由另一个对话推进。第54～60节FC16、joint和center-only均未通过联合probe目标，已结束同缓存展平FC增量搜索。第61～64节几何监督/像素及冻结B空间核对已完成，尚未确定唯一根因。第65～66节旋转ROI四边中点对照工程通过、联合probe条件未通过；保留B ep24，不放行正式训练。用户随后授权尝试TEST，第67节独立固定入口的服务器结果已回传，第68节核对原matched200步全部结果精确一致、保存头state及1440帧逐帧/汇总正确。只执行一次TEST推理，12组打印是同预测的两方法×六分组汇总。real中心/平均RIoU和sim尺寸/角度/平均RIoU有局部收益，但real角度/连续性及sim中心有代价，real_seq04最长RIoU失败3→4帧；不形成联合收益、不替换B。第66节不建议新增TEST是当时建议，授权以第67节为准，原probe失败不被TEST撤销。浅层C2信息对照仍仅拟议，尚未实现；继续问题沿用TEST前第66.5节设计，不据TEST调参/选型，当前P3缓存上的头/损失搜索结束。不保存本地服务器回传副本、一次性复算脚本或源码压缩包，可靠性源码由另一对话维护。**
>
> **当前结论：保留 SymEOOD＋尺度增强 B，VAL固定epoch24，作为后续可靠性研究的暂定检测前端。D 和固定E-H v1均未形成联合收益。E-H在VAL选epoch22，改善real覆盖/连续性及sim角度，但尺寸/RIoU退化；固定TEST上real全帧中心正确数与B相同、输出多1帧，real RIoU基本持平，sim角度及RIoU退化，不替换B。独立尺寸候选F-S v1的保存框数学与服务器有限TRAIN接入/初始化检查已完成；第34节已准备正式配置和VAL比较入口，固定beta/lambda=0.1、原15条件＋普通real中心mean/RMSE保护。2026-10-02已读取F-S完整24epoch训练日志与VAL扫描：原规则选epoch18、唯一可行，real覆盖/RIoU局部改善，sim RIoU低于B，不替换B；已记录显存峰值恒为3007MiB。完整17项VAL缓存比较已复核：7项失败，两域短边与sim纯角度改善，但长边/中心及sim重叠退化；普通real新增4个严重错位，不替换B。固定F-S ep18 TEST已回传并复核：real少21个输出、正确中心总数少26、最长RIoU失败4→9，sim角度/重叠退化；保留B，F-S不作为最终方案，最新分析见第44节。可靠性当前范围、完整实验历史与后续接续见[独立可靠性交接](reliability_handoff_20261003.md)。不加入DINO，不据已多次暴露的TEST调参或重选权重，不恢复候选排序或完整审计。**

当前数据、已完成实验与论文素材已集中整理为[港口新数据与尺度增强实验总记录](detection/港口新数据集与EOOD_SymEOOD尺度增强实验总记录.md)。该主记录维护成果与论文口径，本文保留执行交接及后续设计。早期E经过预检后改为E-H；E-H已完成正式训练和VAL/TEST，F-S已完成前两项有限检查，阶段与结果以本文最新记录为准。

2026-10-03按用户新授权，额外保存[四中点几何修正midpoint v1的方法与结果快照](detection/旋转ROI四边中点几何修正_midpoint_v1_方法与结果_20261003.md)。包含公式、实现、TRAIN/probe及完整固定TEST、原始文献差异与论文主张；后续执行仍更新本文。用户希望保留该候选并研究恢复real DFR/ACI，本轮只记录目标和创新性判断，尚未微调或启动C2/时序代码。详见第69节。

2026-10-01文档整理补充：主记录第3节按实际代码说明SymNFL/SymKLD/SymPOLA及EOOD主头损失接入，第7节并入本文第15节的E机制、文献边界与验证门槛。本文公式中的`global_positive_count`指当前GPU batch跨FPN层汇总的正样本归一化（逐图至少计1），不是跨GPU all-reduce。此次只整理文档，未修改训练代码或重新运行实验；EOOD配置声明L1但本地主头未返回该项的发现，不能未经服务器历史版本核验就外推到已有训练。

> **可靠性记录分离（2026-10-03）：** 原第35～40节、第41节“结构/分量质量分支”、第43/46/47节及R1.1～R1.22已逐字迁至[可靠性交接](reliability_handoff_20261003.md)。本文保留迁出节编号与跳转；F-S的另一第41节及所有几何结果/设计继续保留。原记录中的“同文”或跨节引用按新旧两份索引解析；可靠性后续只更新新文件。

## 1. 新窗口先读这部分

### 1.1 用户目标

用户希望完成小论文：以 EOOD 为原方法主基线，说明 SymEOOD 在真实港口检测覆盖与连续性方面的价值，并改善当前平均定位精度、仿真域角度精度及尺寸精度的代价；之后再评估预测框输入下的测量/深度可靠性。

早期目标是轻量化大 DINO 并保留其能力，尝试过 DINO→SymEOOD 蒸馏和融合。但用户目前明确：**暂不考虑后续 DINO，先把 SymEOOD＋尺度增强的几何精度做好。** 不为了论文复杂度强行加入 DINO。

2026-10-03文档分工：当前B上的可靠性目标、轴线/分支实验、TRAIN/VAL/TEST证据及最新“先完成简单三个判断”的决定已完整迁入[独立可靠性交接](reliability_handoff_20261003.md)。本文继续维护检测几何优化；可靠性旧节位置保留跳转，不在两个文件重复追加实验记录。

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

## 28. B/E-H正式TRAIN日志结果评审与代码复核（2026-10-01）

### 28.1 证据与独立复算（事实）

用户提供终端汇总及`/Users/mac/Downloads/port_shape_e_h_v1_train_log_review.json`。报告SHA256 `7957cb4d6e79baf15f369c59a5d59809fa9d28e5774543ff5600509073614b7f`，原字节归档`work_dirs/port_shape_e_h_v1_train_log_review_server_20261001.json`。其中tool_sha256与本地第27节原工具一致。本轮读取报告、工具源码及必要的MMCV日志/优化器和SymPOLA源码；没有连接服务器、修改工具/模型、推理、训练或读取新增TEST。

E原日志为`20261001_135113.log.json`，SHA `9761fa6f830afef835801d367a256f9d271e4f0dd3162cfb3422d9a03921e745`；B为`20260929_155706.log.json`，SHA `17ea018fe0167c38001b6e96562c0666128126895c95549c4c32f249108a85d2`。两臂各288条TRAIN行，epoch1–24每epoch12行，iter50/100/.../600，无重复/倒退；各排除1条元数据和12条VAL记录。源报告记录seed0、各自正式实验名和metadata issues为空。原始日志及完整config文本未回传，原日志SHA与完整训练配置只能作为服务器记录身份，不能冒称本地独立重算元数据。

独立按回传的576条数值轨迹复算每条辅助loss合计、已加权H/KLD比、总loss与组件和之差，24个epoch/6个固定窗口的n/mean/median/p90/min/max、>10窗口均值计数、全部B/E比较比值，均与JSON一致，也与终端六位有效数字一致。无记录缺失/非有限事件，主分类/KLD和E形状项没有日志零值。组件和最大偏差两臂均约3e-5，与日志舍入/浮点累加的量级一致；没有漏掉形状项或重复乘0.05的统计证据。

每epoch实际训练契约640 iter，日志仅到600，末40步未列入这类50步间隔输出；因此288条日志不是全部训练step的逐步轨迹，不能以epoch覆盖证明每步均有限或训练完整性。窗口平均也会隐藏单层/单步归零保护、个别尖峰。该限制与原工具说明一致，不为补齐日志重训。

### 28.2 初期数值疑点与后期结果（事实）

首窗口（epoch1 iter50）：E主分类2952.19232、辅助分类309.40352、KLD1.37577、H0.01468、grad_norm461044.57015；B分别2952.19095、309.40425、1.37810、无H、461045.88508。两者都具有同量级初期大分类/大梯度，但这是50步窗口均值，不是第50步的瞬时值。

到第二窗口（iter100），E主分类0.35275、辅助分类0.22597、grad_norm120.08476；B 0.35354、0.22896、118.02817。首窗口极大值没有持续到后期。epoch1两个模型12个窗口均值都>10；epoch2–4 E有5/36、B有6/36；epoch5以后均无>10窗口均值，最大值E4.51952、B4.48621。不能把上述计数称为实际step裁剪频率或声称后期完全不裁剪。

后期17–24共96窗口：

| 日志标量中位数 | B | E-H |
|---|---:|---:|
| 主分类loss | 0.015965 | 0.016670 |
| 主KLD loss，已含权重2 | 0.015985 | 0.015790 |
| 新增H loss，已含权重0.05 | 无 | 0.001435 |
| 辅助分类合计 | 0.001200 | 0.001220 |
| 辅助框回归合计 | 0.000330 | 0.000300 |
| 裁剪前范数窗口均值的median | 1.81834 | 1.82086 |
| 上述范数p90 | 1.94590 | 1.96151 |
| 上述范数max | 2.23536 | 2.15266 |

E相对B主分类median高约4.42%、主KLD低约1.22%、辅助分类高约1.67%、grad_norm差约0.14%，仅为固定窗口描述，不能宣称显著差异/分类根因；B/E训练batch/正样本/优化路径未逐位配对。E总loss包含新增项，不能直接将总loss较高解释为更难收敛。

E后期23–24主分类0.014235、KLD0.014045、H0.001360，继续下降；形状/KLD逐窗口比的median由epoch1 1.4396%、2–4 4.0538%、5–16 5.5542%、17–22 9.0281%到23–24 9.4606%，17–24总体9.1057%。这是各日志行比值的median，不等于两个median相除，也不等于梯度比或系数动态增大。H绝对值同时降低，原KLD降得更快；不同压缩/平滑响应和中心/形状误差组成都可能影响该比值，未读取逐框T分布，不能定量归因。

E shape_positive_count为epoch1–3每窗口18，epoch4–6逐渐下降，epoch7起每窗口2；这是每卡记录/日志窗口统计，不是整个DDP batch仅2个正样本。每卡2图/每图1GT、SymPOLA topk9（前2000 iter）→接下来2000 iter渐减→topk1的已有代码，与该趋势一致。不能把18→2写成检测漏失或监督突然消失的证据。B没有该计数字段，不能直接比较两模型的正样本组成。

日志LR在epoch5–16为0.0025，17–22为0.00025，23–24显示0.00003。MMCV TextLoggerHook `_round_float`对float保留5位小数，原配置0.0025×0.1×0.1=0.000025在该写法下round(...,5)=0.00003。因此末段显示3e-5不构成学习率配置错误，也不能从舍入日志反推精确LR；实际配置值仍2.5e-5。

### 28.3 代码审查结论与一项边界缺口（事实、待修复）

**当前输入的窗口分组、loss归类、权重口径、分母、p90、比较与裁剪描述复核正确，没有发现会改变本次数值判断的统计错误。** `review_required=false`只表示所实现的数据/元数据检查没有标记缺口，不是训练稳定性、全协议或几何收益通过证书。

静态核对发现一项工具元数据健壮性缺口：`metadata_review`仅在`resolved`非空时核对必需配置字段。若config字符串有效但不含任何目标赋值（例如`unrelated = 1`），而seed/exp_name符合预期，当前实现会返回identity_review_required=false。标准库最小复现已确认。未来应对空resolved明确标记缺失，并保存所核对的关键配置摘要，避免只能看到哈希/成功flag而无法本地复核。这属于需要修补的输入身份检查边界；回传数值统计独立复算不依赖该分支，但完整原config未提供，不能用当前flag替代独立训练配置审计。

用户本轮要求读取/检查/分析，未将只读审查自动扩为改代码：该缺口记录为待修复，没有擅自改工具或要求重跑本次已核实数值报告/训练。现有报告继续保存。修补这一分支不会提供新的训练数值或解释E几何退化。原主头非有限归零与窗口舍入的可观测性限制也保持，不把有限日志当作从未触发保护的证明。

### 28.4 本次测试目的是否完成与有限下一步（推断、待验证）

目的1（初始大分类/梯度是否长期持续）：现有日志支持其主要为初期现象，后期没有相同量级持续异常；不能再以初始化探针的大梯度直接解释已选E-H权重的几何退化。目的2（E项是否接入且保持非零标量）：日志支持，形状项参与组件和且全程非零；不代表每层/每框都有有效梯度。目的3（是否存在可见后期损失/窗口范数崩溃）：未见，B/E后期相近；不能升级为完整每步数值稳定性证明。目的4（为何普通框尺寸与sim几何退化）：尚未解决，标量日志没有宽高/角度梯度分解。

**下一步建议是有限TRAIN梯度检查，不是新训练或系数扫描。** 沿用既定两对TRAIN图×clean/0.5等比例缩小，分别加载冻结B ep24/E-H ep22，固定真实后期O2O/分类温度状态，零optimizer step；复用实际主KLD正样本节点及权重/分母。记录有符号长短边误差、正样本身份/数量及几何保护，检查主KLD和H对log宽高（w*dL/dw、h*dL/dh）及角度的直接梯度与当前误差的纠偏方向，再看主回归卷积/FPN的相对强度和夹角。不能直接比较px宽高梯度与rad角度梯度的原始范数来宣称哪项支配；也不能把B/E不同正样本的梯度当完全匹配的配对样本。

若确需辨别中心—形状耦合，仅在同一实际节点上用已有C/S数学分解作有限描述，不引入新损失或改变分配。检查可检验当前小样本上H是否有尺寸纠偏响应、是否与主KLD交互，但不能单凭一个方向/夹角认定全部训练根因。B仍为有效方案；既定等比例、原图还原、深度约束和无DINO/无完整审计路线保持。新增设计仅基于TRAIN/VAL，当前TEST已多次暴露，不继续调权/选epoch或新增TEST推理。

## 29. 冻结B/E-H有限TRAIN梯度检查实现、复核与运行指令（2026-10-01）

### 29.1 授权、固定范围及实现（事实）

用户要求按第28.4节建议继续审查、完成代码复核并给服务器指令。本轮在原项目目录新增`crane_project/tools/diagnose_port_shape_e_h_train_gradients_v1.py`、来源指纹清单`crane_project/tools/port_shape_e_h_train_gradients_v1_sources.json`及对应测试；修补第28.3节日志工具缺口。没有连接服务器、启动训练/优化器或新增VAL/TEST推理。B仍是有效方案；E-H原公式、0.05系数、正式训练配置、选权和已完成结果均保持。

梯度工具严格固定B ep24/E-H ep22，读取原VAL sweep的身份字段并核对配置SHA、选中路径、权重SHA及第24/28节已记录的权重指纹，不重新选择候选或使用指标打分。服务器加载权重后，安全AST解析已保存config的model/data/optimizer/optimizer_config/lr_config/runner/load_from/resume_from，与当前已解析配置逐项比较；验证seed0、epoch24/22、iter15360/14080。没有执行checkpoint内的config文本，也不重新ImageNet初始化。读取整份sweep JSON仅为身份核对，不读取预测缓存或将其中VAL指标用于本次诊断。

只用既定TRAIN索引(0,1810)/(905,2184)，同时验证四张图的准确名字：real_seq01_00000、sim_seq08_00000、real_seq06_00006、sim_seq08_00374。两种视图clean/0.5，各权重4个batch，共8次真实训练模式前向；每batch两图、每图一个GT，固定late O2O，不重查早期O2M、不扩成全TRAIN审计。每batch重置相同的Python/NumPy/torch随机种子，核对B/E实际输入张量、GT、图片SHA、scale_factor和尺寸完全一致；不同尺度输入不是像素等价视图，缩小的细节不可逆。两臂TRAIN数据计数固定1810+748，报告保存两域标注集合SHA。

模型进入原有TRAIN/norm_eval状态。探针临时设置主分配器计数及全部相关SymNFL计数到后期，核对分类tau=tau_min；断言每图一个正样本、每层使用原全局分母2。Python分配器计数不在state_dict中，不能只加载权重就假定已处于后期。每batch结束或异常都恢复计数和所有buffer；任何非调度buffer变化立即拒绝，训练中的BN也立即拒绝。每臂全部检查结束再次核对参数SHA和所有buffer字节，不更新参数、不调用backward/clip/optimizer step。上轮窗口范数已核实，本次不模拟新裁剪轨迹。

### 29.2 检查内容及解释口径（事实、推断边界）

复用实际`head.loss_bbox.forward`收到的正样本预测/GT节点，核对逐层原主KLD之和与输出主loss一致。E-H额外项须共用相同预测/GT对象，权重数值与分母相同，且与固定0.05公式独立计算值一致。权重在现有主头中重新切片，产生相同值但不同对象，故检查权重值而不错误要求其Python对象相同。空正样本层也要求可追踪的实际节点，梯度不能因截获兄弟节点而全部None后被当成零梯度。用于几何描述的拼接均detach，绝不将拼接兄弟节点作为autograd目标。

每个正样本保存图片、level/anchor身份、预测/GT、权重/分母及主KLD/H的有符号梯度响应：长短边log误差、对应`w*dL/dw`/`h*dL/dh`（按预测长短边映射）、误差×梯度，以及规范长边方向的周期角度误差、角度梯度和乘积。误差×梯度>0表示冻结分配下对解码框变量的无穷小SGD纠偏响应；它不是一次实际参数更新或训练收益证明。尺寸梯度和角度梯度单位不同，分别报告，不通过其原始范数大小排名。保存GT/预测长宽比、近方形方向可辨识性和角度周期边界标记，避免将宽高交换/近方形角度当成无条件错误。

共享层检查主回归输出卷积、分类输出卷积及整个FPN的主KLD/H/base_total范数、同组H/main与H/base比及夹角；base_total包含原辅助loss。H对实际框中心及直接分类输出卷积的梯度必须为零，但共享层变化仍可能间接改变中心、分类和覆盖。B上的0.05 H仅是冻结B节点的反事实梯度参照，明确标为counterfactual，不是B已训练过的损失；E上的H为实际原额外损失。不建议匹配梯度自动调系数，也不输出新系数。

保存decoder宽高delta限幅、解码边/中心边界、主KLD float32行列式/逆矩阵/原距及压缩上限，以及额外H float64的独立保护计数，不能用H的double保护结果证明KLD无保护。同一图片B/E可能选择不同anchor，报告逐图标记是否相同；不能将不同正样本的梯度当严格配对差异，也不重新开启候选排序路线。没有加入C/S梯度分解或另一种loss：当前主KLD/H直接方向已经是本轮补的关键缺口，若结果仍不足再论证有限分解的必要性。

原图描述按TRAIN现有RResize的反变换：中心除sx/sy，两条边除sqrt(sx*sy)，其中scale_factor已包含共同的0.5缩小；兼容keep_ratio整数舍入造成的sx/sy微差。核对clean/half的GT还原相同、scale_factor恰为共同0.5。此处仅生成TRAIN正样本诊断坐标，不改推理坐标还原或深度链路，也不宣称测量了独立深度精度。本次不计算TRAIN正样本“检测中心命中率”；检测报告仍须遵守输出帧中心命中、输出覆盖及全帧中心正确覆盖三项分母口径。

### 29.3 日志边界修复、代码复核和本地验证（事实）

日志工具现对空/部分resolved配置标记`resolved_config_required_fields_missing`，将所解析的model/runner/optimizer_config/work_dir写入`checked_config_summary`供复核。标准库回归测试验证空config目标字段不再误报身份通过、正常配置摘要及不可执行config；没有改日志数值算法/裁剪口径，没有覆盖第28节原结果。之前“待修复”是历史状态，本节已完成修复；修补不提供几何根因或新训练数值。

本地Python3.8/torch1.8 CPU执行相关三组测试共27项通过：6项新梯度测试、13项日志测试、8项原E-H集成测试。覆盖真实SymEOODHead/实际节点和空层、log边及角度有限差分、宽高交换/周期等价、角度纠偏、近方形限定、带舍入的原坐标还原、完整且安全的checkpoint配置核对、冻结权重身份与错误拒绝、计数/钩子异常恢复、BN和非调度buffer修改拒绝、8批次CPU I/O fixture/B-E输入配对及状态不变、CLI成功/失败报告、来源不匹配和拒绝覆盖。CPU fixture替代ResNet/FPN输入路径、服务器权重和CUDA接口，不能视为正式权重或GPU数值证据。

独立配置命令完成，报告`work_dirs/port_shape_e_h_v1_train_gradients_local_config_20261001.json`状态为`CONFIG_ONLY_RUNTIME_UNVERIFIED`。Python AST、diff空白及31个上传成员逐字节校验通过。来源清单固定28个实际依赖文件的SHA，服务器执行前强制校验，执行后再次校验；包含为既有helper配置核对所需的D配置字节，不加载D权重或运行D实验。除了两个诊断工具及测试/清单/本记录，未修改模型、损失、分配、增强、正式配置、推理/深度代码。

上传包`work_dirs/port_shape_e_h_v1_train_gradients_tools_20261001.tar.gz`含工具/测试/清单及已核对的既有代码依赖，共31文件，无数据、权重、日志或评估缓存；依赖文件与当前项目原字节相同。SHA256 `a500ce0ea26a03f4e410aea9f53705d23c6430c70474e5e94867d3277ec8ae49`。新梯度工具SHA `c958a49afb673399b5aada190280beee8f82247f5954d73de99553010b33de9b`。未连接服务器，正式CUDA/实际权重数值和显存仍待验证。

### 29.4 服务器命令、回传及后续判据（待验证）

把上述包上传到项目根目录后运行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
tar -xzf port_shape_e_h_v1_train_gradients_tools_20261001.tar.gz

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/diagnose_port_shape_e_h_train_gradients_v1.py \
  --gpu 0 \
  --out-json work_dirs/port_shape_e_h_v1_train_gradients.json
```

物理卡3映射到进程gpu0。默认使用B/E原`val_sweep_port_v1`目录；只有原sweep位置迁移时用`--b-sweep`/`--e-sweep`指出原身份文件位置，不能换epoch/权重。清单不匹配须同步包内已核对代码，不绕过清单/配置/权重检查。已有输出或任何sidecar存在时拒绝覆盖；复用旧产物或用新输出名，不删除重要结果来重跑。

回传终端及以下三个文件：

- `work_dirs/port_shape_e_h_v1_train_gradients.json`：完整检查报告。
- `work_dirs/port_shape_e_h_v1_train_gradients.progress.jsonl`：逐batch检查记录，正常8行。
- `work_dirs/port_shape_e_h_v1_train_gradients.artifacts.json`：GPU执行前冻结身份/来源。

正常完成状态`TRAIN_GRADIENT_CHECK_COMPLETE_REVIEW_REQUIRED`只表示诊断完成需人工分析，不是新实验获准、几何改善通过或根因证明。报错时主JSON保留CHECK_FAILED、progress保留已完成batch，先根据错误及原证据复核；不自动改batch/尺度/公式/系数，也不因显存失败改实验身份。

仅为补齐旧日志元数据摘要，可选复读同两份原日志，不占GPU、不提供新轨迹：

```bash
python crane_project/tools/analyze_port_shape_e_h_train_logs_v1.py \
  --e-log work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/20261001_135113.log.json \
  --b-log work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/20260929_155706.log.json \
  --out-json work_dirs/port_shape_e_h_v1_train_log_review_metadata_v2.json
```

后续分析先看保护/实际节点/配对身份，再结合有符号误差与梯度判断四张TRAIN图上H是否纠偏尺寸/角度、是否与主KLD同向或有代价、是否通过共享层影响原目标。单个相反夹角、标量比或微小单次变化不能证明全训练根因、监督权重不足或优化器实际更新效果。仅在TRAIN机制证据与既有VAL退化模式一致时预登记下一项范围有限的设计/对照，继续保留B。TEST已多次暴露：不以本轮或既有TEST差异调系数、改选权、改门槛或启动新TEST。

## 30. 冻结B/E-H TRAIN梯度结果评审与有限下一步（2026-10-01）

### 30.1 回传证据、身份及独立复核（事实）

用户提供第29节命令的8行终端打印和Downloads下主JSON、progress JSONL、artifacts JSON。原字节归档到`work_dirs/port_shape_e_h_v1_train_gradients_server_20261001.json`、同前缀`.progress.jsonl`和`.artifacts.json`，未覆盖服务器/之前报告。三者SHA分别为：

- 主JSON：`bf5316450298b1dfe53c90b77560a887ee2fb84b81300649bb97f4d5e8971ff8`。
- progress：`c0dbdf22600a114da030be3e9c007cda2be33cd52b33968b45a064fb7e5f3b9e`。
- artifacts：`714ef4429adb10dc8edc9423ef92fe9c3b45c5ecb18ae76bd3412a69f1304e61`。

主JSON状态TRAIN_GRADIENT_CHECK_COMPLETE_REVIEW_REQUIRED；progress 8行与主JSON的8个row逐字段完全一致；artifacts是执行前快照，CHECK_STARTED属于预期状态，除status外全部字段与最终主JSON对应前缀一致，不是未完成检查。8行终端H和FPN H/main数值与JSON逐项完全一致。

本地独立核对28个源码SHA、来源清单SHA及B/E配置SHA与当前项目一致；两个权重SHA与第24/29节已冻结B ep24/E-H ep22一致。服务器checkpoint_contract均MATCH、seed0、iter15360/14080，参数/buffer未改变、optimizer_steps=0、分类tau1、late O2O每图1个正样本/分母2。权重本地仍缺失，checkpoint读取/参数不变结论为服务器已核对代码的报告证据，不能声称本地再加载权重。

四张TRAIN原图SHA及train/train_sim标注集合SHA与本地数据一致；两臂同视图的输入/GT/scale_factor/image SHA一致。B/E 8个图片—视图配对中6个level/anchor相同；clean第一对real_seq01_00000和sim_seq08_00000不同，因此这两个不能视为严格匹配正样本比较。即使另外6个anchor相同，两臂参数、优化历史及epoch24/22仍不同，不能升级为只切换H的因果干预。

保存框的raw主KLD范围0.00345936–0.05600375，decoder边/中心边界及delta限幅、主float32行列式/逆矩阵/原距/压缩上限、H double保护/指数尾部计数均0。GT长宽比约1.5010、2.7041、2.9142、2.9239；全部方向定义/周期边界标记通过，本次不是近方形框或角度周期端点造成的符号异常。服务器峰值allocated显存两臂约2.296GiB，仅表示此次探针，不是完整训练显存预算。

独立按报告norm复算全部比值，按框/梯度复算误差×梯度及几何响应，均一致。进一步仅用回传的16个正样本框，在本地CPU重算原SymKLDLoss/权重2和固定H/权重0.05，使用相同权重1和分母2：batch主loss最大绝对差1.1921e-7，H为1.1642e-10；log长/短边及rad角度梯度最大绝对差主KLD9.2387e-7、H为0。这是保存节点的数学复算，没有运行检测器、推理、optimizer或增加TRAIN图片。结合第29节已通过测试与此次源码/数值复核，未发现改变当前判断的节点、权重、分母或统计错误；FPN/卷积夹角没有原梯度向量和本地权重，不能声称本地独立重新算出这些夹角。

### 30.2 新增H强度与共享层交互（事实、推断边界）

以下H已包含0.05，主KLD已包含权重2，不再次乘权重：

| 冻结模型/视图 | FPN H/main范数比（两batch） | FPN H/main cos（两batch） |
|---|---:|---:|
| B clean，反事实H | 11.0899% / 11.1199% | 0.5507 / 0.9438 |
| B half，反事实H | 4.9335% / 4.2540% | 0.5586 / 0.9175 |
| E-H clean，实际H | 7.2835% / 7.6411% | 0.9266 / 0.8619 |
| E-H half，实际H | 4.7953% / 4.9170% | 0.3522 / 0.9766 |

E主回归输出卷积H/main比clean6.3119%–6.7952%、half3.1773%–4.7081%，其cos全部正（0.2203–0.9589）。E的FPN H/base_total比4.7925%–7.5993%、cos0.3504–0.9689；base_total含原主分类与辅助项。H对直接分类输出卷积和实际框xy均零，主回归卷积base_total梯度与主KLD一致，符合分类质量分支detach和独立辅助输出卷积的原结构。

有限批次中H在主回归/FPN有非零响应，没有整体反向共享梯度或数值限幅失效的证据；但positive cos不能保证每张图/每个尺寸或角度都纠偏，也不能保证共享参数更新保持中心/分类/检测覆盖。half的H/main比更低并不证明0.05太小：本次E H的绝对FPN范数clean约0.0632/0.0777、half0.0694/0.0742，量级相近，main的范数和预测误差组成也改变。不能将这些比值当作形状监督“充分/不足”的判据或直接据此增大系数。

### 30.3 直接纠偏响应与一个局部耦合实例（事实、限定推断）

每臂8个正样本，定义误差×梯度>0为冻结分配下局部纠偏：B反事实H与原主KLD对长、短、角均8/8；E实际H与主KLD对长和角均8/8、短边均7/8。只有E half的sim_seq08_00374短边例外；16个框来自4个原图及重复视图/权重，不能当作16个独立样本或纠偏率的总体估计。

该框level0/anchor13146，B/E选择相同anchor。E当前长边+6.9694%、短边+0.5558%、规范角度误差-3.4825°，原图中心距2.7259px；已偏大的短边，其log短边梯度主KLD=-0.0050239、H=-0.0002380，局部梯度下降会继续增大短边。H角度梯度-0.0201104，与负角误差相乘为正，角度正在被纠正。不能将短边不纠偏写成整体协方差loss上升或无效；联合几何目标可能同时降低其他误差。

为补这一具体缺口，在同一保存框上做float64解析反事实，保留尺寸，分别仅把中心或规范方向替换为GT，再重算同样loss的log短边梯度，没有新模型前向、修改训练目标或实际更新：

| 保存框的局部反事实 | 主KLD log短边梯度 | 0.05 H log短边梯度 |
|---|---:|---:|
| 实际框 | -0.00502469 | -0.000237993 |
| 仅中心=GT | -0.00415730 | -0.000237993 |
| 仅方向=GT | +0.01042363 | +0.001023962 |
| 中心和方向均=GT | +0.01103533 | +0.001023962 |

同时在无保护区用已有C+S表达分解原主KLD、保留原sqrt压缩链式因子：raw C=0.00323824、S=0.03557227，对log短边梯度贡献分别约-0.000873874和-0.004150818，总和-0.005024692。仅把中心置为GT不消除例外，H不依赖中心且仍为负；只消除角差即转为纠偏。因此可将这个单框的符号变化解释为尺寸—方向联合项的局部耦合，并指出主中心项也贡献扩边响应，而非代码断连、周期或数值保护。

这不是D失败根因，也不是E-H全局退化根因：例外短边是偏大的，而既有VAL主要记录整体偏小；直接用这个反例解释VAL尺寸低估在方向上不成立。此外，E half的real_seq06_00006短边低估-9.0095%（B为-6.5179%），当前H和主KLD却均有纠偏梯度；其中心/角度又比B好。说明“保存点的方向正确”与“不同训练轨迹最终尺寸更好”是不同证据，不能由局部导数推出收敛、共享网络响应或泛化。

### 30.4 当前结论与下一步建议（推断、待验证）

**有限TRAIN检查完成，E-H的实际接入和局部几何响应得到验证；没有找到足以解释全局退化的训练根因。保留B；E-H v1仍为未达联合目标的固定实验。** 结合第28节后期日志，没有证据支持继续以持续大梯度、H完全无效、整体梯度反向或数值保护触发作为既定根因；也不直接提高0.05、重训原E-H或恢复D/候选排序/完整审计路线。

下一项可优先考察的有限候选是**原B＋独立长短边log尺寸约束**：保留B原主KLD、中心/角度和全部检测链路，仅令新增项直接约束log(pred_long/gt_long)、log(pred_short/gt_short)，不新增H、中心或角度项。尺寸排序应兼容宽高交换，目标detach，尺寸无量纲；原等比例变换、坐标还原、深度约束不变。此为待设计/验证的机制假设，检验“单独尺寸纠偏是否比继续增加联合协方差监督更匹配既有VAL尺寸低估”，不是已证明的新loss或收益，不能承诺同时改善角度/覆盖。

先用现有TRAIN保存框及已归档VAL普通尺寸低估框做有限离线公式/梯度检查，关注上下偏差、角差、近方形、交换等价和本节例外；不新增推理或全数据审计。根据该TRAIN检查预先固定公式、一个系数及预算，不能按VAL/TEST网格挑系数。通过后才另行获授权在本地实现一项B/候选同初始化、同预算对照，按原规则选权，使用第15/24节覆盖、连续性、尺寸、定位、sim角度及RIoU联合门槛，不事后放宽。原B/E实验结果继续保留；若有必要重新训练B作同期对照须事先说明，不以小样本反事实证明收益。

本次只读取三个新TRAIN附件、已有交接/相关代码和旧VAL报告结构，未重算新VAL全量梯度或读取新增TEST，未改任何诊断/模型/损失/配置代码。TEST已多次暴露，继续明确披露；后续设计、系数和选权只依TRAIN/VAL协议。本次不报TRAIN正样本检测“中心命中率”，也没有输出覆盖或全帧中心正确覆盖的新证据；这些检测分母和深度独立验证限制保持。

## 31. 独立尺寸候选F-S v1的详细设计与针对性文献核对（2026-10-01，尚未实施）

### 31.1 任务边界及问题对应（事实与推断）

用户要求先详细设计下一步并说明是否针对现有问题，再检索相似工作。本节先形成设计，再以原论文核对其依据及局限；只更新本文，没有改代码、连接服务器、启动训练/推理或新增TEST检查。B ep24继续为有效方案，D/E-H历史结果和选权身份保持。F-S是新候选代号，不把已失败的固定E-H v1改名或改变其历史公式。

| 当前证据（第24/28/30节） | F-S的直接目标 | 不能由此推出的结论 |
|---|---|---|
| VAL两域长短边signed log bias为负，E-H比B更负；普通real尺寸也退化 | 对每条边的尺寸比例提供独立纠偏信号，同时约束偏大和偏小 | 不证明主KLD是整体偏小的根因，也不是所有框都偏小 |
| 一张TRAIN框的主KLD/H短边导数与该边误差方向不符，消除角差后转为纠偏 | 新增项的边长导数不依赖角差/中心差 | 例外是短边偏大，不能用于解释VAL整体低估；原主KLD耦合仍保留 |
| E-H改善sim角度、real正确覆盖，但尺寸、RIoU及定位尾部有代价 | 回到B，只增加尺寸项，检验不同监督目标的取舍 | F-S不直接新增角度/中心监督，不保证复现E-H角度或覆盖收益 |
| 后期日志没有持续大梯度；有限TRAIN H有非零且多数纠偏的响应 | 无证据要求先改lr/clip或继续增大H系数 | 正确的局部导数不证明共享网络更新、收敛或泛化更好 |

**工作假设：** 在保留原联合几何目标的条件下，额外的独立边长纠偏能否改善尺寸误差及尺寸偏差，并满足原覆盖、定位、角度和连续性要求。它针对尺寸这个已观测缺口；不是整体根因修复，也不是完全解耦网络。

静态核对发现，`mmrotate/models/losses/center_size_compensation.py`中的D尺寸分项已经是排序边长log残差的两维平均SmoothL1，beta=0.1；D另含中心分项，二者合计权重0.25。因此F-S的数学主体并非本项目首次引入，其实验身份是**在B上单独检验尺寸分项**。本节候选权重0.1与D也不同，B/F-S对照不能单独归因D失败于中心项；如要严格分解D因果，需要另设同系数消融，但本轮不增加该路线。

### 31.2 一个具体候选公式及接入范围（设计，待TRAIN验证）

对原主KLD实际接收的解码正样本框，定义排序边长：

```text
Lp = max(pred_w, pred_h); Sp = min(pred_w, pred_h)
Lg = max(gt_w, gt_h);     Sg = min(gt_w, gt_h)
eL = log(Lp/Lg);          eS = log(Sp/Sg)
rho_beta(e) = e^2/(2*beta)               if abs(e) < beta
              abs(e) - beta/2           otherwise
g_i = (rho_beta(eL_i) + rho_beta(eS_i))/2
L_F-S = L_B + lambda * sum_i(a_i*g_i)/N
candidate: beta=0.1, lambda=0.1, eps=1e-6
```

`a_i`与`N`必须直接复用原KLD的实际权重和当前GPU batch跨FPN层归一化，逐层使用同一全局分母、再求和；不增加跨GPU归一化、按域重加权或每层自行平均。GT detach。空正样本返回与预测边相连的零项，负样本不参与。沿用原O2M/O2O切换和原正样本，不增加样本筛选、IoU质量加权、候选排序或难例挖掘。

新增项只读取真实解码的w/h，使用与D一致的两维平均，不能将求和/平均混用造成2倍差别。正边数值检查和eps只保护log；不另加上限/偏小专用权重、输出尺度校准或小框放大。原解码限幅仍须诊断，不能用解码边上的导数证明限幅后的原参数仍有梯度。尺寸排序消除w/h交换编码影响，角度周期不进入新增项；不改变GT角度定义。

候选beta=0.1沿用既有D尺寸项平滑尺度，二次区对应约-9.52%到+10.52%的尺寸比例误差。候选lambda=0.1采用一个温和、可解释的起点：无保护、中心和方向完全正确、尺寸残差很小时，项目raw对称KLD约为`2*(eL^2+eS^2)`，权重2及sqrt压缩后的主回归约为`2*(eL^2+eS^2)`；新增项约为`lambda/(4*beta)*(eL^2+eS^2)`，故此理想局部条件下额外尺寸强度约为原项的12.5%。这是本项目公式的泰勒近似，不是论文验证，不是实际FPN范数比，也不是选择了最优系数。

**0.1/0.1是本节提交评审的唯一候选，尚未通过有限TRAIN接入/初始化检查，也尚未冻结正式训练合同。** 后续检查若数值、节点或有效信号失败，先报告具体失败并重新论证，不自动扫系数、不按VAL/TEST挑系数、不临时加入阶段调度。小样本范数比不用于自动匹配H强度。

现有`SymEOODHead.shape_compensation`通过通用`build_loss`接入，且已传递主KLD实际正样本节点/权重/分母；后续可复用此入口，新建独立尺寸loss和继承B的配置，不新增预测分支或可学习参数。原日志键`loss_shape_compensation`若复用，报告必须另明确loss类型为F-S、公式/系数及配置SHA，避免与H混淆。需有关闭新增项时所有原loss、state键及推理输出一致的检查。历史D/E-H类和正式配置不改，F-S中不叠加D或H。

### 31.3 纠偏性质与已知反例的界限（数学推导，未运行F实验）

在排序分支确定、合法正边且无保护激活时：

```text
d L_extra / d log(Lp_i) = lambda*a_i/(2*N) * clip(eL_i/beta, -1, 1)
d L_extra / d log(Sp_i) = lambda*a_i/(2*N) * clip(eS_i/beta, -1, 1)
d L_extra / d pred_x/y/theta = 0
```

因此新增项对有误差的边总有独立纠偏方向；对预测和GT的共同等比例缩放不变。两条边同时缩小但比例正确时仍会有loss，避免只约束长宽比遗漏共同尺度。近方形时不人为加强方向监督；精确w=h的排序拐点须验证一致的分支/次梯度，不能声称处处可微。

**原主KLD加上新增项的总梯度未必纠偏。** 对第30.3节保存的E half反例，仅代入候选公式作标准库数学运算：短边log残差0.005542746，权重1/分母2下F-S导数为+0.001385687；原主KLD约-0.005024691，两者相加仍为-0.003639005。这不是在B上运行F，也不是新训练结果；它表明候选提供纠偏补充，而不保证翻转该保存框的联合导数。不能因未翻转就按此单框增大系数，也不能宣称消除了角度耦合。

共享回归塔/FPN及预测相关分配仍可能使分类、中心和覆盖变化。保持等比例增强、原图坐标还原和原深度估计约束；F-S不新增推理变换。代码链路保持不等于独立深度精度已验证，几何或尺寸改善也不能直接替代深度GT评估。

### 31.4 有限验证、正式对照与判断（待实施）

1. **先复用保存框作数学/单元验证。** 用第30节16个保存节点检查上述导数、有限差分及主KLD+F-S组成；覆盖边长上下5%/10%、两边共同缩放、中心/方向替换、w/h交换加pi/2、theta加pi、近/精确方形、空正样本和数值保护。角差2/5/10度等是数学扰动，不能算新数据或训练证据。已归档VAL普通低估框仅用于固定的几何解释，不拟合系数/门槛、不重跑全量推理。不得把独立构造的邻近张量梯度当实际主节点梯度。
2. **授权实现后做有限TRAIN接入/初始化检查。** 复用原四张TRAIN图及clean/half，不扩大为完整审计。B ep24探针仅用于后期实际节点/尺度/共享层梯度；另按原seed0和ImageNet初始化核对B/F-S参数与buffer一致、O2M真实接入和完整目标裁剪。冻结后期快照强制早期分配不等于恢复早期历史。检查新增项xy/theta直接梯度为零、边长/回归/FPN非零、原loss一致、保护和clip记录完整；所有探针optimizer_steps=0、状态/钩子/计数恢复。是否出现有效信号须在实际解码及限幅链路下判断，不能只看loss非零或新增项与主项正cos。
3. **检查通过且正式设计冻结后，一次B/F-S对照。** F-S从与B相同ImageNet预训练/seed0开始，不从B或E权重续训；TRAIN real1810+sim748、等比例增强、24epoch、原两卡总batch4、SGD/lr日程及clip10均不变。优先复用原B日志/权重/VAL缓存；核对来源、初始化和配置合同，仅新增loss/工作目录不同。历史B是单seed历史对照，不冒称同期逐batch配对。若来源/初始化/环境实质不一致，先披露并说明同期B重跑的必要性，不自行增加训练预算。
4. **沿原VAL协议选权，再判断目标。** 仍只扫描16/18/20/22/24，原硬约束与软评分、fallback披露不变；不能按尺寸报告换epoch。第15/24节15项覆盖/联合几何门槛全部保留：real输出至少374/375、全帧中心正确至少360/375，最长无输出/RIoU失败不劣于1/4；sim输出与中心正确保持512/512；两域共同输出长/短边平均相对误差都下降、sim纯角度RMSE严格低于B实际2.107151度、共同输出中心mean不增、全帧平均RIoU不降。中心命中仅以输出帧为分母，另报输出覆盖及全帧中心正确覆盖；不得把B的96%全帧正确写成100%。

普通real沿用原固定排除10个历史严重错位帧的分组，另报其尺寸误差、中心mean/RMSE/p90、signed bias和逐序列新增严重错位。本节建议正式冻结前明确增加**普通real共同输出中心mean/RMSE不得恶化**的定位尾部保护项（B现有共同组n364基准4.4375px/5.2923px；新增输出另报，不能混换分母）。此项是拟新增目标保护，不伪称原15项已包含；不参与选权评分、不在看到F结果后改标准。共同输出集合若改变，应同时重算该集合的B基准，不硬用不同集合的上述数值。

如果仅尺寸改善、sim角度不改善，或定位/覆盖代价仍在，就报告部分收益并保留B；不为通过实验放宽联合目标。单次seed0或微小差异不称稳定/显著收益。TEST已经多次暴露，不能称未接触的独立确认集；本节不依据TEST调公式/系数/选权，也不安排新TEST。

### 31.5 针对性文献证据—设计对应表（已检索事实与可借鉴边界）

检索日期2026-10-01。主题词覆盖rotated/OBB regression、size-angle coupling、independent shape/side-length loss、scale sensitivity和gradient calibration。使用原论文arXiv全文、NeurIPS官方PDF、期刊发布页及作者所在机构文献库；不把第三方摘要或论文AP当本项目证据。是针对性检索，不是系统综述/穷尽原创性检索。RIL/EIoU读取方法部分，Constraint/2026 RSFPN核读发布页方法内容，KLD复用并核对原文，GCL仅依据作者机构可读摘要，SODE依据发布页摘要/引言；未读的GCL完整推导不能写成已经复现。

| 原论文及核读位置 | 相似问题/可借鉴内容 | 本候选采用或不采用、证据边界 |
|---|---|---|
| Xue Yang等，[Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence](https://proceedings.neurips.cc/paper/2021/file/98f13708210194c475687be6106a3b84-Paper.pdf)，NeurIPS2021，参数梯度与尺度不变性分析 | KLD的参数梯度相互影响，并可随物体形状调整优化重点 | 保留联合主KLD；耦合也可能有益。论文没有验证本项目整体低估根因或lambda=0.1 |
| Qi Ming等，[Optimization for Arbitrary-Oriented Object Detection via Representation Invariance Loss](https://arxiv.org/html/2103.11636)，2021预印本，第3.3节式9/10 | 有不考虑中心/角度的尺度不敏感shape度量，并考虑宽高等价表达 | 最接近独立尺寸的OBB依据。借鉴表示等价/相对尺度；不引入其完整匹配策略，也不照抄完整RIL |
| Yi-Fan Zhang等，[Focal and Efficient IOU Loss for Accurate Bounding Box Regression](https://arxiv.org/html/2101.08158)，Neurocomputing2022，第3.1.3/3.2节式5–7 | 仅长宽比可遗漏两边共同缩放，且比例项可能使宽高响应相反；EIoU显式约束两条边 | 支持分别测量边长误差。它研究水平框/CIoU，不证明KLD或H有相同全局故障；不引入Focal样本加权 |
| Luyang Zhang等，[Constraint Loss for Rotated Object Detection in Remote Sensing Images](https://www.mdpi.com/2072-4292/13/21/4291)，Remote Sensing2021，DOI10.3390/rs13214291，第2.1节式3 | 把中心、尺寸、角度分开，尺寸对宽高交换作最小误差处理 | 支持独立尺寸和交换不变性已有先例；其重点是表达边界/约束容差，不等同当前退化。不采用中心/尺寸小误差置零的容差域 |
| Qi Ming等，[Gradient Calibration Loss for Fast and Accurate Oriented Bounding Box Regression](https://biblio.ugent.be/publication/01HYJQR0WF0MQZ32KWVZ89D2J6)，IEEE TGRS2024，DOI10.1109/TGRS.2024.3367294，机构摘要 | 对rotated IoU梯度和角误差关系、尺度敏感性作分析及校准 | 借鉴实际梯度验证，不能只看loss下降。其角误差与梯度负相关不等于错误符号，也不是主KLD/H问题证明；暂不换GCL |
| Xiaozhi Yu等，[SODE-Net: A Slender Rotating Object Detection Network Based on Spatial Orthogonality and Decoupled Encoding](https://www.mdpi.com/2072-4292/17/17/3042)，Remote Sensing2025，DOI10.3390/rs17173042，摘要/引言 | 尺寸位置与角度特征分支解耦，并改backbone/角度编码 | 提供未来共享特征冲突的研究方向，超出单loss受控范围；目前没有本项目特征冲突的充分证据，暂不引入 |
| Jiaxin Xu等，[Rotation-Sensitive Feature Enhancement Network for Oriented Object Detection in Remote Sensing Images](https://www.mdpi.com/1424-8220/26/2/381)，Sensors2026，DOI10.3390/s26020381，第3.4.3节式17/18及消融说明 | 联合回归另加log长宽比形状项，按历史梯度统计动态平衡权重 | 相近的“原回归+形状项”先例；但仅比例项对两边共同缩放为零，不能替代本候选两条边的尺度纠偏。不引入其动态权重、角度项或FPN改造 |

对RIL式10作本节自己的代数改写：固定边长对应关系时，`hIoU = exp(-abs(log(wp/wg))-abs(log(hp/hg)))`。因此该度量与log边长差有直接关系；F-S的SmoothL1在小误差附近的响应不同，不是RIL原式。式10是论文定义的shape度量，不能随意把它等同一般二维矩形真实IoU或未经核对直接照搬论文中loss符号。

对2026 RSFPN式17的本节数学判断：若pred_w=k*gt_w且pred_h=k*gt_h，长宽比相同，其shape项为零，虽然k不为1时实际尺寸有误差。发布页还明确记载其消融中加入GC-MTL后总体mAP相对上一阶段下降0.14个百分点；这仅是原论文该指标的结果，不能证明其所有几何效果无效，也不能把整套网络AP收益归为形状项。因此“有论文做形状监督”不等于这种监督解决本项目共同尺度偏小；动态权重不由论文名义自动获得采用依据。

**文献核对后的设计建议：** 继续采用B加独立log长短边项这个最小候选，优先借鉴RIL的表示/尺度处理、EIoU的独立边长目标和GCL的梯度核查方法；不把仅长宽比损失当完整尺寸监督，也不一次加入动态权重、角度分支或新FPN。已找到相似工作和相似优化问题，没有找到直接验证本项目F-S公式/权重及联合目标的结果，更不能宣称“新OBB损失”或原创性。正式验证仍按31.4节，收益和根因均保持待验证。

## 32. F-S前两项检查的本地实现、验证及服务器交付（2026-10-01）

### 32.1 授权范围与实现（事实）

用户授权综合第31节建议与相关论文，进行前两项检查、本地修改对应代码，复核后提供服务器指令。已实现候选F-S和有限检查工具；没有连接服务器、正式训练、修改VAL选权/门槛或新增TEST。原B ep24仍为有效方案。候选beta=0.1、lambda=0.1、eps=1e-6固定用于本次检查，但**尚未根据服务器实际接入/初始化结果冻结正式训练合同**。不把代码可训练或检查完成状态当成正式训练许可。

| 项目位置 | 本次内容 |
|---|---|
| `mmrotate/models/losses/log_size_loss.py` | 排序长短边log残差的两维平均SmoothL1，GT detach；没有中心/角度依赖或参数 |
| `mmrotate/models/losses/__init__.py` | 仅注册/导出LogSizeLoss |
| `crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1.py` | 继承B，仅新增尺寸项和工作目录，未叠加D/H，未改初始化/预算 |
| `crane_project/tools/check_port_size_f_s_v1.py` | CPU保存框解析梯度/有限差分、表示/尺度/角差与方形边界检查，无检测器前向 |
| `crane_project/tools/preflight_port_size_f_s_v1.py` | 冻结B后期4批次＋原seed0初始化4批次的TRAIN接入/梯度/完整裁剪检查，零optimizer step |
| `crane_project/tools/port_size_f_s_v1_saved_boxes.json` | 既有16个TRAIN节点和8个固定描述性VAL低估框；记录来源SHA及原TRAIN身份 |
| `crane_project/tools/port_size_f_s_v1_sources.json` | 34个代码/配置/保存框成员的固定SHA |
| `tests/test_port_size_f_s_v1.py` | 12项F-S数值、真实主头接入、状态/异常恢复和CLI契约测试 |

数学主体沿用已有D尺寸分项。本轮借鉴RIL的交换等价/相对尺度思路、EIoU的独立边长目标及GCL的实际梯度核查，不声称复现完整论文；未加入动态权重、角度分支、样本筛选、候选排序、蒸馏或FPN改造。模型主头、解码、分配、增强、坐标还原及深度代码未改。旧D/E-H损失/配置、原诊断报告未覆盖。

### 32.2 第一项：保存框公式检查结果（本地事实，数学证据）

从第30节原TRAIN梯度报告提取全部16个保存节点，保留原框、权重、分母及arm/尺度/图像/anchor。另从第24节原VAL比较报告提取每arm/domain按图像名排序的前两个两边均低估且非原10帧严重错位的输出框，共8个；这是明确条件选择的描述性例子，不是无偏样本、总体纠偏率或新VAL评估，不据此挑系数。来源报告及当前fixture SHA均固定。

本地Python3.8/torch1.8 CPU执行结果保存为`work_dirs/port_size_f_s_v1_saved_box_check_local_20261001.json`，状态`SAVED_BOX_FORMULA_CHECK_COMPLETE_REVIEW_REQUIRED`。包括16个TRAIN回放、8个VAL描述框及34个合成比例/角差/方形扰动。无检测器前向、推理或参数更新。

新增项的解析log边梯度与autograd一致；所有保存框新增xy/角度导数为零。表示交换加pi/2、theta加pi、中心/方向改变和共同0.5缩放不改变新增项；两边共同偏小仍被检测。近方形/精确方形有限，精确排序拐点明确不作不适用的双侧有限差分。全部非拐点log边有限差分最大绝对误差约1.36724e-7，低于预先设置2e-6。

| TRAIN保存框回放 | 原主KLD局部纠偏（长/短） | 单独F-S局部纠偏（长/短） | 主KLD＋F-S局部纠偏（长/短） |
|---|---:|---:|---:|
| B，8个节点 | 8/8、8/8 | 8/8、8/8 | 8/8、8/8 |
| E-H，8个节点，只作历史反例验证 | 8/8、7/8 | 8/8、8/8 | 8/8、7/8 |

重复视图/模型来自4张原图，不能把这些分母当独立样本或全TRAIN纠偏率。E-H half的sim_seq08_00374短边导数：原主KLD -0.00502469137，F-S +0.00138568647，合计 -0.00363900490，仍不纠偏。与第31节限制一致；没有通过增加系数消除此反例，也没有借它解释VAL共同尺度低估。

首轮核查发现并修正了两个**诊断断言问题**：B配置的完整loss字典还含eps/reduction，不能以省略字段的字典误拒；当前MMDetection的weight_reduce_loss在mean/avg_factor路径含float32 eps分母保护。新增loss一直调用原reduce函数，未改归一化。解析检查现在测量同版本reduce倍率，再验证边长导数，报告记录倍率和库版本。当前本地N=2时float64倍率0.4999999702、float32倍率0.5；第31节1/N公式是忽略此微小库保护的数学表达，正式接入继续与原KLD共用原归一化。

### 32.3 第二项：TRAIN工具契约及本地可验证范围（事实、服务器待验证）

工具依赖第一项成功的报告并核对候选、源码清单、fixture及状态，拒绝换系数、跳过前置检查或覆盖已有输出。GPU前冻结artifacts，逐批写progress，失败保留CHECK_FAILED及已完成行；没有自动改batch/尺度/预算或启动训练。

服务器只载入原B ep24，核对原VAL选权身份/config SHA、已冻结checkpoint SHA和checkpoint内安全AST配置/seed/epoch/iter；不读取E/D权重、不用VAL指标重选权。旧sweep JSON仅用于身份字段。B权重中新增F-S为反事实信号，不能当已经训练F。之后按tools/train.py的seed0→build_detector→init_weights核对当前B/F-S参数/state键/buffer完全相同，再使用F-S初始化模型作实际O2M检查；不从B续训，也不把当前初始化身份相同写成已恢复历史初始状态。

每阶段固定TRAIN索引(0,1810)/(905,2184)，即real_seq01_00000、sim_seq08_00000、real_seq06_00006、sim_seq08_00374，clean/half各两批，共8个带F-S的前向。每批额外关闭F-S作一次原B目标前向，因此**实际共16次TRAIN检测器前向**，不是只有8次；全部无optimizer step。两阶段输入张量/GT/scale_factor/图像SHA一致；half原GT还原与clean一致且共同尺度恰为0.5。TRAIN仍为real1810+sim748，图像字节与两域标注集合SHA必须匹配原报告，不仅检查名字/数量。checkpoint执行后再次验SHA。

新增项复用主KLD实际正样本张量对象、权重及每层全局分母，捕获并核对所有5层、空层和loss计数；独立公式与集成值一致。记录主KLD/F-S及二者相加的直接长短边响应、xy/theta为零、正样本level/anchor、输入和原图坐标。检查回归输出卷积/FPN有效梯度，新增分类输出卷积和回归输出xy/angle行梯度为零；共享梯度夹角不作为几何收益保证。记录主KLD原距、det/inverse/raw/loss保护、原decoder限幅、尺寸eps/线性区/排序tie，不能把F-S无保护当主KLD无保护。

完整B目标包括原辅助项，分别对完整B与B+F-S反传并按原clip10记录前后范数/倍率；没有模拟optimizer更新。比较关闭F-S后的原全部loss时恢复相同RNG和调度状态，防止随机辅助分支产生假差异。所有钩子、计数、buffer、RNG及参数保持/恢复；BN训练态、非调度buffer修改、实际节点断连或非有限梯度均拒绝，异常后清除临时参数梯度。

这里是单GPU、每批两图的有限探针，保持原每卡batch/分母；完整目标指该批次全部原loss，不代表重构了正式两卡DDP平均后的历史梯度、裁剪或更新轨迹。

本地12项新测试通过，相关26项原尺寸/E-H集成及梯度测试通过，共38项。覆盖上述数值/归一化/GT detach、D尺寸分项等值、真实SymEOODHead实际节点/空层、原loss和推理不变、参数/state键一致、随机辅助loss比较、完整裁剪、错误/BN/buffer拒绝、异常梯度清理、八批次CPU I/O fixture、初始化匹配、前置报告/源码拒绝及CLI持久化。CPU fixture以小卷积替代ResNet/FPN输入链路、CUDA及checkpoint读取，不能称服务器真实B权重、原ImageNet参数或GPU数值已通过。

独立配置命令完成，`work_dirs/port_size_f_s_v1_config_check_local_20261001.json`状态`CONFIG_ONLY_RUNTIME_UNVERIFIED`。本地库为torch1.8.0.post3/MMCV1.7.2/MMDetection2.28.2；服务器报告会记录其库版本、reduce源码SHA和倍率。Python3.8 AST及diff空白检查通过。源码清单/上传成员另逐字节核对；除新增F-S相关文件及loss注册外，没有修改历史模型/损失/分配/数据转换代码。旧E-H源码清单保留原历史字节，注册新增loss导致当前__init__.py SHA改变；本次使用独立F-S清单，不重写历史清单消除版本差异。

### 32.4 服务器运行与回传（待验证）

上传`work_dirs/port_size_f_s_v1_preflight_tools_20261001.tar.gz`到服务器项目根目录，再执行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
tar -xzf port_size_f_s_v1_preflight_tools_20261001.tar.gz

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/check_port_size_f_s_v1.py \
  --out-json work_dirs/port_size_f_s_v1_saved_box_check.json

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_size_f_s_v1.py \
  --gpu 0 \
  --math-report work_dirs/port_size_f_s_v1_saved_box_check.json \
  --out-json work_dirs/port_size_f_s_v1_train_preflight.json
```

第一条是CPU数学检查，可复现本地结论；第二条仅执行有限TRAIN接入/初始化检查。物理卡3映射gpu0。B原sweep目录迁移时，仅用`--b-sweep`提供原目录；不换权重/epoch，不绕过来源或数据身份检查。原ImageNet初始化沿用服务器torchvision预训练缓存。本地没有该B权重/GPU环境，实际数值、初始化/裁剪及显存待服务器回传确认。

回传终端和以下4个文件：

- `work_dirs/port_size_f_s_v1_saved_box_check.json`。
- `work_dirs/port_size_f_s_v1_train_preflight.json`。
- `work_dirs/port_size_f_s_v1_train_preflight.progress.jsonl`，正常8行。
- `work_dirs/port_size_f_s_v1_train_preflight.artifacts.json`，GPU前身份快照。

预期末尾分别为`SAVED_BOX_FORMULA_CHECK_COMPLETE_REVIEW_REQUIRED`和`TRAIN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED`，都只是检查完成待评审，**不是正式训练或联合几何目标通过**。artifacts的CHECK_STARTED是执行前快照，不代表主报告失败。输出或sidecar已存在时改新输出名，保留旧结果；若改数学输出名，第二条math-report同步引用它。

后续先看实际节点/来源/保护、初始及后期的尺寸有效信号、共享响应与完整裁剪，再决定是否冻结一次正式F-S训练；不由非零loss、正cos或小样本比例直接自动调系数。第31.4节VAL协议/覆盖、普通real尾部及联合几何目标继续有效，尚未改VAL工具。TEST已多次暴露，不用于本次设计/系数/选权；TRAIN正样本不是输出帧，不在本工具报告检测中心命中率/输出覆盖或独立深度精度。

### 32.5 最终交付校验（事实）

上传包共39个成员：34个固定来源成员、独立F-S清单及4个相关测试文件；没有图片、权重、完整训练日志、完整VAL预测或TEST产物，只有12KB的既有框诊断fixture。包大小139060字节，全部成员与当前项目逐字节一致，Python3.8 AST及diff空白检查通过。原第30节28个来源中27个仍保持原SHA，唯一变化是本节明确新增导出的loss __init__.py。包不含本文，项目交接继续以本地本文为准。

- 上传包SHA256：`8cd055c580a2e4876927ffef8b0e4452d6adf5838415dd78e3ba9d1205cddb63`。
- F-S来源清单SHA256：`4e27e775f3bfe3abbb4ab93ed4111b57532fe3dded55b018497908dd8354ec7f`。
- 本地保存框报告SHA256：`2087ea50b522281bec26d1faccdb1db31de4fb0a92beccc97acf2d556c581743`。
- 本地配置报告SHA256：`90b4fae920cec029cfd8ba994c3cd2fb50faf48c81846561de43c5a54701e0c1`。

实现/测试复核未留下影响本次数学或接入检查的已知代码问题。第一项已在本地完成；第二项完成工具、真实主头CPU验证及配置核对，服务器真实权重、ImageNet初始化和CUDA运行仍待回传，不作“GPU已通过”或“可正式训练”的结论。

## 33. F-S服务器前两项检查结果评审与正式对照建议（2026-10-01）

### 33.1 回传、身份与独立复核（事实）

用户要求读取终端与4个结果文件，分析是否推荐正式实验对照。本轮只读核对相关实现、复算保存数值、归档回传并更新本文；没有改模型/训练/评估代码、连接服务器、重跑检测器、启动优化或TEST。第32节的服务器待验证事项现有实际回传；本节判断替代其当时的“尚不能判断正式训练”状态，不改写历史检查结果。

以下文件按原字节归档到`work_dirs/`，没有覆盖原报告：

| 归档文件 | SHA256 |
|---|---|
| `port_size_f_s_v1_saved_box_check_server_20261001.json` | `56ae0ad8f6f8f58992a549771b30628636881b0f0c8c3b67180a4aa3b05ee49d` |
| `port_size_f_s_v1_train_preflight_server_20261001.json` | `c5f49d28732343d6b744f3b9b446cfbe4fe0d214eea381d835dfa37c38c0ca5d` |
| `port_size_f_s_v1_train_preflight_server_20261001.artifacts.json` | `5cdd82d8d2e5f96af70bbf232840104615a26ec798bd5debca0345b226ee69fe` |
| `port_size_f_s_v1_train_preflight_server_20261001.progress.jsonl` | `7a11df8a629f80c6e1255e7f3a42126c0208f32242e00b4947cd8251b3b365df` |
| `port_size_f_s_v1_preflight_terminal_server_20261001.txt` | `8ff7884a2be30a6829f9b0f257abc6f4e49232f02536b0f405d4e084f3aef35b` |

主报告状态为`TRAIN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED`，数学报告为`SAVED_BOX_FORMULA_CHECK_COMPLETE_REVIEW_REQUIRED`；终端8行数值逐项匹配主报告，progress正常8行且与主报告rows完全相同。artifacts的`CHECK_STARTED`是GPU前快照，身份字段与主报告一致，不能把它解释为没有完成。`formal_training_authorized=false`保留了探针运行时的执行范围，不是接入失败或后续实验价值的自动判定。

34个回传源码SHA与当前清单/项目一致，清单SHA仍为`4e27e775f3bfe3abbb4ab93ed4111b57532fe3dded55b018497908dd8354ec7f`；数学报告SHA与TRAIN前置绑定一致。候选仍为LogSizeLoss/beta=0.1/loss_weight=0.1/eps=1e-6/mean。B身份仍为原VAL选epoch24，checkpoint SHA `8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23`；checkpoint元数据检查model/data/optimizer/clip/lr/runner/load/resume、seed0/epoch24/iter15360匹配，没有换权重。

TRAIN仍real1810＋sim748，固定4张图、clean/half各两批、两阶段共8批；实际执行16次检测器TRAIN前向（每批另关闭新增项），optimizer_steps=0。标注集合/4张图SHA匹配前次证据，两阶段输入张量、GT、metadata相同，half共同尺度及原图GT还原一致。后期每图1个正样本、每卡全局N=2；初始化每图9个、每卡全局N=18，5个FPN层都复用该分母，不能按层自行平均或把它称为两卡总数。后期O2O/tau=1，初始O2M/tau=9.991（一次前向后计数）沿用原调度，初始两个模型参数/state/buffer相同。两阶段参数/buffer均保持或恢复，B文件执行后SHA不变。

实际物理GPU0映射gpu0，GTX1080；torch1.13.1+cu117/MMCV1.7.0/MMDetection2.25.1，reduce源码SHA与先前本地核查一致。两阶段峰值约4.28/4.30GiB，仅为单卡两图的诊断峰值，不能推断完整两卡训练预算或历史DDP梯度。当前B/F-S初始化一致也不能替代缺失的历史初始参数逐位比较；正式比较应披露历史B对照身份。

### 33.2 数学、实际尺寸导数与保护（事实、机制边界）

保存框数学检查覆盖16个原TRAIN节点、8个固定条件选择的描述性VAL框及34个合成扰动，共58例。最大非拐点有限差分误差1.36735e-7，低于预设2e-6；交换/周期/中心角度改变/共同尺度不变性误差至多1.12757e-17，共同尺寸偏小可被识别，方形排序拐点沿用分支导数政策。与本地已通过报告相比，主/尺寸loss最大差4.44e-16；不用重复同版本完整测试。8个VAL例子不是无偏抽样或几何改善证据。

独立使用回传输入框、权重及分母复算全部80个实际正样本：F-S标量最大差3.21249e-9，log长短边导数最大差7.57492e-8，主KLD＋尺寸导数的相加误差至多1.65310e-8，均在现有float32容差内。归一化、范数比、裁剪倍率另行复算一致。复核结果为`work_dirs/port_size_f_s_v1_preflight_review_local_20261001.json`，SHA `32cdbe620139f1bd1c9da554b3bde14ec4e650d6363ac8eed114a5ae3c56764b`；此复算无检测器前向/优化，未独立重建完整FPN梯度向量，夹角仍来自已绑定源码的服务器报告。

新增项的实际xy/theta直接导数、分类输出卷积梯度、回归输出xy/angle行梯度均为0，边长/回归卷积/FPN梯度均非零；关闭新增项时全部原loss保持一致。8批decoder尺寸delta限幅、边/中心裁剪、主KLD det/inverse/raw/loss保护、负raw、Gaussian边保护，以及尺寸eps保护/tie均为0。初始144条边中120条在SmoothL1线性段，这是预设大残差响应，不能当数值限幅故障；后期16条边均在二次段。

| 实际节点 | 原主KLD纠偏（长/短） | 单独F-S纠偏（长/短） | 主KLD＋F-S纠偏（长/短） |
|---|---:|---:|---:|
| 冻结B后期，8个 | 8/8、8/8 | 8/8、8/8 | 8/8、8/8 |
| 原初始化，72个 | 53/72、56/72 | 72/72、72/72 | 53/72、56/72 |

这是4张原图的重复尺度/正样本视图，不是80个独立图像或全TRAIN纠偏率。初始总项仍有19条长边、16条短边导数不纠偏，本次F-S没有翻转它们；后期原KLD本身已全部纠偏。因此本结果支持“独立纠偏信号已接入”，不能声称“修复了所有原梯度错误”或“已证实尺寸偏小根因”。第32节E-H保存反例仍未被候选总项翻转；未依单框改系数。

### 33.3 后期信号与初始化裁剪（事实、推断）

P1=real_seq01_00000＋sim_seq08_00000；P2=real_seq06_00006＋sim_seq08_00374。以下比值为同批FPN参数梯度范数比，loss值已含0.1系数；cos是新增尺寸与完整B目标的FPN夹角余弦。

| 阶段/批次/尺度 | F-S loss | 尺寸/主KLD FPN | 尺寸/完整B FPN | cos尺寸/完整B | 完整B裁剪前范数 |
|---|---:|---:|---:|---:|---:|
| 冻结B P1，1.0 | 0.000208554 | 5.700% | 4.887% | 0.3757 | 2.20680 |
| 冻结B P2，1.0 | 0.000631553 | 11.793% | 11.732% | 0.9385 | 1.29292 |
| 冻结B P1，0.5 | 0.000415661 | 3.697% | 3.569% | 0.3453 | 5.43601 |
| 冻结B P2，0.5 | 0.001566467 | 5.728% | 5.607% | 0.4876 | 4.13899 |
| 初始化 P1，1.0 | 0.024231814 | 2.320% | 0.0003222% | 0.00308 | 1212340.375 |
| 初始化 P2，1.0 | 0.025464281 | 2.904% | 0.0009855% | 0.00488 | 396018.59375 |
| 初始化 P1，0.5 | 0.044037815 | 1.280% | 0.00007515% | -0.05008 | 4001913.25 |
| 初始化 P2，0.5 | 0.034859154 | 1.461% | 0.0001530% | -0.00159 | 3786760.75 |

冻结B后期，尺寸/回归输出主KLD范数为1.92%–10.30%，尺寸/主KLD FPN为3.70%–11.79%，尺寸/完整B FPN为3.57%–11.73%。与主KLD及完整B的FPN cos分别0.3634–0.9403、0.3453–0.9385。完整B范数1.293–5.436，B＋F-S为1.420–5.479，四批均不触发clip10。**推断：** 在这些真实解码/共享层节点上，信号有效且未被完整目标裁剪，足以支持一次受控收益实验；正cos不保证普通框尺寸、覆盖或时序提高。

初始化新增项相对主KLD FPN为1.28%–2.90%，相对完整B FPN仅7.51e-7–9.85e-6（无百分号的比例）。主分类loss3566.9–30230.6、辅助分类152.3–2790.3，完整梯度被既有分类目标主导；裁剪倍率2.50e-6–2.53e-5，裁剪后均约10。B/F-S初始裁剪前范数相同或仅相差0.25/4.00e6，不能据总范数相同说新增梯度为零。两个half批次与完整B的cos略负，说明与主KLD正cos不能替代完整目标响应。

**限定推断：** 初始尺寸信号受到共同强裁剪，早期有效更新很弱；此现象同时存在于原B，既有B/E-H日志首窗口的大分类/裁剪与之相符且后来下降（第28节，窗口均值不等于逐step）。因此它不是本次F-S特有的数值失败，目前没有证据要求单独改F-S的lr/clip、增系数或加调度。零step探针不能预测新训练多久收敛，也没有证明初始状态理想或整个训练稳定。

### 33.4 正式对照建议与固定评价条件（建议、待验证）

**推荐进入一次固定F-S v1正式对照，不继续扩大预审。** 数学、实际节点、数值保护、状态恢复和后期共享信号已完成必要检查，没有发现会改变本次判断的接入/统计代码问题；新收益需要训练/VAL才能回答。此建议没有自动启动训练，也不是宣布F-S替换B。回传证据不足以证明最优系数、联合精度改善或根因。

建议正式设置沿用当前唯一候选：B＋独立排序log长短边SmoothL1，beta=0.1、lambda=0.1、eps=1e-6，复用主KLD正样本/权重/当前GPU全局分母；不叠加D/H。从原ImageNet/seed0初始化，24epoch、两卡每卡2图（总batch4）、SGD/lr日程/clip10、TRAIN real1810＋sim748和等比例增强均保持；不从B/E续训，不扫系数或添加动态权重。优先复用匹配身份的B ep24/日志/VAL缓存，仅新增一次F-S训练；这是单seed历史B对照，不能称同期逐batch配对或显著/稳定改善。

正式运行前把F-S的VAL比较身份入口准备齐全：现有`ckpt_sweep.py`可沿原配置入口处理F-S，但`compare_port_shape_e_h_val_v1.py`绑定E-H配置/候选身份，不能只换目录冒充F-S报告。本轮未修改评估工具；后续有限适配应复用原缓存/指标和门槛，核对F-S配置SHA及选权身份，不另做全量审计。这是评价交付缺口，不是需要重跑当前TRAIN预检的理由。

原VAL候选16/18/20/22/24、硬约束/软评分/fallback及15项联合门槛不变，不根据尺寸/后期loss重选epoch。建议在F结果产生前把第31.4节提出的普通real共同输出中心mean/RMSE不得恶化一并固定，作为新增尾部保护，不参与选权评分；共同输出集合变化时重算同集合B基准，不能直接套用旧n364的4.4375/5.2923px。逐序列严重错位和连续性仍按原定义另报。

real中心命中仅按输出帧统计，并单列输出覆盖率/全帧中心正确覆盖；B现有VAL是输出374/375、中心正确360/374、全帧正确360/375，不能写成全帧100%正确。两域长短边误差、signed bias、中心mean/普通real尾部、RIoU及sim纯角度RMSE共同判断；只尺寸改善而角度不改善或覆盖/定位/时序有代价，应报告部分收益并继续保留B，不放宽原目标。维持等比例变换、原图坐标还原及原深度约束；深度精度仍待独立GT检验。

沿用正式训练既有每50iter日志观察分类/辅助分类、主KLD、新增尺寸loss及裁剪前grad_norm。初始共同大值不自动判失败；如NaN/Inf/步骤错误，定位具体数值问题；如大分类/极端裁剪持续不下降，优先查共同初始化/分类链路，不边看结果边改F-S系数、lr或评价条件。当前TEST已多次暴露，本轮不安排新TEST，不根据它设计、调参或重选权重。先完成固定TRAIN/VAL对照，再评审是否值得下一阶段。

## 34. 固定F-S正式配置、VAL入口、代码复核与服务器指令（2026-10-01～2026-10-02）

### 34.1 授权及实现（事实）

用户授权按第33节建议修改项目、复核代码并提供服务器指令，明确不打压缩包；中途询问显存后要求继续。本轮新增以下文件，保留上一轮本文改动和所有历史源码/配置/报告；没有连接服务器、正式训练、真实模型VAL推理或TEST。

| 文件 | 内容 |
|---|---|
| `crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1_formal.py` | 独立正式入口，完整继承已验证的F-S候选，不改任何解析后的训练/模型/推理设置 |
| `crane_project/tools/compare_port_size_f_s_val_v1.py` | CPU正式合同检查，及原B ep24/原规则F-S所选VAL缓存的只读比较 |
| `crane_project/tools/port_size_f_s_v1_formal_protocol.json` | 在F正式结果产生前固定公式、17项条件、B身份/原10帧分组及41个相关源码SHA |
| `tests/test_port_size_f_s_val_v1.py` | 身份/选权/分母/尾部/元数据/缓存来源/文本一致性/异常拒绝验证 |

原候选`size_f_s_v1.py`保留创建时的预检注释及SHA。正式入口虽为新路径，但`Config.to_dict()`与候选完全相同（工作目录也相同）；仍只有相对B新增LogSizeLoss，beta=0.1/lambda=0.1/eps=1e-6/mean，不叠加D/H、不续训、不改初始化/预算。原34成员预检清单SHA仍`4e27e775f3bfe3abbb4ab93ed4111b57532fe3dded55b018497908dd8354ec7f`，旧数学/TRAIN报告继续有效，**不需重跑GPU预检**。

正式入口SHA `441aa8ace4621c9eda00a80b764c76c53b82106e73f1987a4bc96c761e8ff758`；新比较工具SHA `37ee574a29011eee497dcc45f61e98b688e5cd2cd88d3ad1be59157a99e00382`；冻结协议SHA `734fa74aa4a38025bf3e682af63576d1e2a66ddce307ec9f208406a909d9a0d2`。以后VAL必须使用正式入口对应的配置SHA，不能换回候选配置路径假称相同缓存来源。

### 34.2 评价合同及代码审查（事实、待验证边界）

CPU `--check-only`核对固定源码、候选与正式解析配置等值、原已评审TRAIN报告SHA/状态/参数/零step、B身份和当前TRAIN标注SHA；没有模型构造、权重读取或GPU调用。新增`--require-reviewed-library`可在服务器拒绝与已评审预检不同的torch/MMCV/MMDetection及reduce合同；本地库不同，静态验证仅记录该差别，不伪称当前本地已等价服务器。正式训练仍为原ImageNet/seed0、24epoch、两卡每卡2图、SGD/lr/clip10、real1810＋sim748等比例增强。

完整比较只读取B固定epoch24和F-S原VAL规则的已选epoch；要求16/18/20/22/24完整候选集合、原选择配置/metric版本/15px阈值、VAL标注SHA及配置SHA相符。B的原选权JSON/权重/PKL SHA固定；两域887条缓存沿原loader读取，验证PKL/TXT、生成provenance以及CPU读取的checkpoint元数据。训练配置字段用字面量AST核对，不执行checkpoint内配置代码；seed0、selected epoch及每epoch640iter必须匹配，不把不同loss或续训权重冒充F-S。

几何分解、共享输出、signed log bias、逐序列无输出/RIoU失败及RIoU独立复核全部复用原函数，原15项门槛原样保留。现将普通real共同输出中心mean/RMSE不得恶化两项正式固定，总计17项；不进入选权评分。排除组固定为原B的10个零RIoU输出帧，两模型在同一共同输出集合上重算基准，另报该集合图名/计数和新增零RIoU输出。没有把旧n364常量套在新共同集合上。中心命中率只用输出帧分母，输出覆盖率和全帧中心正确覆盖单列；real基准360/375不是全帧100%。

比较工具不推理、不写缓存/权重/选权JSON、不重选epoch；输出已有时拒绝覆盖，保留fallback信息、历史单seed对照及TEST已多次暴露等限制。17项通过只表示该次固定VAL满足预先条件，不自动替换B或证明稳定/显著收益及独立深度精度。当前没有F-S正式精度结果。

### 34.3 必要验证与显存判断（事实、推断、待验证）

最终15项新测试全部通过，另2项原选权约束/缺失指标测试通过，共17项相关检查。包含真实887条VAL标注＋合成GT预测下的实际PKL/DOTA/生成provenance/元数据读取和文本篡改拒绝；没有把这些合成预测写成模型效果。历史E-H报告只作为统计回归fixture，原15门槛与原报告逐项相同；其普通real尾部仍失败、B自身尾部等值通过。错误环境、候选缺失、改阈值/配置/标注/seed/保存配置、错序和报告覆盖均拒绝。

Python3.8 AST、源码SHA及diff空白检查通过。首轮新工具的CONTROL导入位置错误在测试收集时发现并修复；最终版本没有留下影响本次范围的已知代码问题。原损失/头/分配/增强/解码与其已通过的38项检查未变化，没有重复完整审计或训练梯度测试。

最新本地合同结果`work_dirs/port_size_f_s_v1_formal_contract_local_20261001_v2.json`（SHA `7b0132b9a294c059ec782ce8982ddf820c70b3a5df686d9071a5d6df65884163`）确认配置等值与TRAIN标注匹配；`library_matches_reviewed_preflight=false`明确表示本地torch1.8/MMCV1.7.2/MMDetection2.28.2与服务器1.13.1+cu117/1.7.0/2.25.1不同。服务器命令启用严格库合同，应为true；失败时先核对环境，不绕过它开始正式实验。初版本本地合同保留，最新冻结来源以上述v2为准。

**显存静态事实：** LogSizeLoss没有参数/新网络分支/额外图像前向，复用原主KLD解码正样本，仅增加N×2的sort/log/SmoothL1和标量归一化；没有跨iteration保存张量/计算图的容器。正式配置不挂诊断钩子、不使用预检的多次autograd或retain_graph流程。图像/padding、backbone/FPN/辅助头、batch及优化器未改。已有预检后期/初始化峰值约4.28/4.30GiB；两阶段不是纯B/F-S显存配对，差值不能当F-S增量。

**推断与待验证：** F-S本身预计只增加少量显存，没有发现导致持续累积或大幅新增特征张量的代码路径；不能保证正式训练峰值等于/低于预检。正式训练首轮会创建SGD动量、DDP通信缓存等，峰值可能高于4.3GiB。本地MMCV TextLogger已核对：`memory`是累计`max_memory_allocated`的MiB值，多卡取最大值；`nvidia-smi memory.used`还包含缓存/context/其他进程，二者不直接相减作为新增loss代价。用首epoch既有日志和另终端GPU观察确认，不为省显存自行减batch或改分辨率破坏对照。

### 34.4 服务器上传、训练和VAL指令（用户自行执行）

服务器已有第32节完整且SHA匹配的预检源码；保持相对路径上传以下**3个新增运行文件**即可，不打包、不上传本地测试/合同报告覆盖服务器预检：

- `crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1_formal.py`
- `crane_project/tools/compare_port_size_f_s_val_v1.py`
- `crane_project/tools/port_size_f_s_v1_formal_protocol.json`

同一个服务器shell先执行下面代码。静态合同失败或F-S工作目录已存在时不启动训练，避免覆盖历史实验；合同报告已存在则保留并先读取，不重复启动新训练。该检查不重跑GPU探针，默认复用已回传的`work_dirs/port_size_f_s_v1_train_preflight.json`（迁移只改`--train-preflight`路径，SHA保持）。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
FS_CONFIG=crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1_formal.py
FS_WORK=work_dirs/crane_symeood_k1_port_day2night_size_f_s_v1

if python crane_project/tools/compare_port_size_f_s_val_v1.py \
  --check-only --require-reviewed-library \
  --out-json work_dirs/port_size_f_s_v1_formal_contract.json; then
  if [ -e "$FS_WORK" ]; then
    echo "已有F-S实验目录，保留产物，本条命令不重复启动训练。"
  else
    CUDA_VISIBLE_DEVICES=2,3 \
    bash tools/dist_train.sh "$FS_CONFIG" 2
  fi
fi
```

若需要实时观察，在另一个终端执行（Ctrl+C结束观察，不影响训练）：

```bash
nvidia-smi -i 2,3 \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5
```

训练完成后，只运行F-S的原VAL选权，再读既有B缓存作比较，不重跑B、不附加TEST开关：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
FS_CONFIG=crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1_formal.py
FS_WORK=work_dirs/crane_symeood_k1_port_day2night_size_f_s_v1
FS_SWEEP="$FS_WORK/val_sweep_port_v1"

python crane_project/tools/ckpt_sweep.py \
  --config "$FS_CONFIG" --work-dir "$FS_WORK" --sweep-dir "$FS_SWEEP" \
  --epochs 16 18 20 22 24 --center-thresh 15 --mcml-limit 5 --gpu 3

python crane_project/tools/compare_port_size_f_s_val_v1.py \
  --require-reviewed-library --f-sweep "$FS_SWEEP" \
  --out-json work_dirs/port_size_f_s_v1_val_compare.json
```

比较默认B目录为`work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1`，仅在原缓存目录迁移时用`--b-sweep`定位；不改写原选权JSON/身份SHA。VAL扫选与比较既有结果继续复用，失败先报具体身份/数值问题，不为通过门槛改协议。训练前合同不含GPU测量；真实F-S训练稳定性、显存峰值和联合精度仍待服务器回传。

回传训练末尾和首epoch含`memory`/分类/KLD/尺寸loss/grad_norm的日志、`port_size_f_s_v1_formal_contract.json`、`$FS_SWEEP/sweep_results.json`及`port_size_f_s_v1_val_compare.json`。先评审固定VAL结果，再决定后续；TEST已经多次暴露，本轮不新增TEST命令、不用它调系数/阈值/选权。

2026-10-02交付复核：41个正式来源与34个历史来源SHA全部一致，新增Python文件符合3.8 AST，以上3个服务器shell代码块通过`bash -n`，diff空白检查通过。同版本已通过的17项相关测试没有无理由重复执行。最终仅新增正式配置/比较工具/冻结协议/测试并更新本文，没有生成新压缩包、修改原模型/损失源码或执行服务器操作。

## 35. 暂定B前端下的可靠性与连续观测研究安排（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-35)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 36. 当前帧分量可靠性：范围修正与文献结合建议（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-36)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 37. 增强方法贡献的候选：图像结构证据与分量扰动监督（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-37)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 38. B缓存与TRAIN结构标签就绪检查、旧轴线恢复（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-38)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 39. 服务器就绪检查回传：B原始缓存通过，旧轴线归档尚未识别（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-39)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 40. 服务器完整轴线复核通过，可进入结构可靠性有限TRAIN预检（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-40)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 41. 冻结B上的结构/分量质量分支与有限TRAIN接入实现（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-41-structure)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 41. F-S正式训练与VAL扫描回传：real局部改善，sim重叠未达标（2026-10-02）

### 41.1 本轮输入、复核与证据缺口（事实）

读取用户提供的`20261002_083108.log.json`及VAL终端。原字节分别保存为：

| 本地归档 | SHA256 |
|---|---|
| `work_dirs/port_size_f_s_v1_training_log_server_20261002.log.json` | `7cf015ccd269b7a01a0f0ac242047c992734678484bd766a19f20ad0963c123a` |
| `work_dirs/port_size_f_s_v1_val_sweep_terminal_server_20261002.txt` | `612cedada982ba8387b209eb679b4308915cf9b6ff5c1e0b2b601c86507da52e` |
| `work_dirs/port_size_f_s_v1_returned_train_val_review_20261002.json` | `0405e2e420af47c77c9e96d6e2f8a5bf0e6eb265400e702a35e82cb5c4289acd` |

复核JSON由本地只读复算生成，不是服务器正式17项缓存比较报告。安全解析日志配置的字面量与`dict(...)`，不执行输入中的配置文本；model/data/evaluation/optimizer/clip/lr/runner/checkpoint/log/load/resume/work_dir共12个字段与当前正式配置一致。seed0、auto_resume=false、两GPU、每卡2、24epoch、ImageNet初始化、原B增强、原主KLD权重2及F-S beta/lambda=0.1均符合设计；没有D/H叠加。日志记录torch1.13.1+cu117、MMCV1.7.0、GTX1080两卡。当前本地41项冻结源码SHA仍匹配；这不等于已取得服务器全部源码原字节。

原`select_best_checkpoint`函数用终端打印的五组数值复算，选epoch18、`constraint_pass`、1/5可行，无fallback，与终端一致。**尚未收到F-S的`sweep_results.json`、正式合同JSON或`port_size_f_s_v1_val_compare.json`。** 因此原始权重/PKL/TXT身份、未舍入指标、17项逐帧几何条件及RIoU数值交叉核对仍待现有比较入口确认，不把终端解读替代该报告。

### 41.2 VAL选权及相对B的结果（事实；F-S暂依终端）

实际数据为VAL887帧：real375、sim512。`[TEST模式]`是既有离线评价器取得完整时序指标的运行模式名；此次命令及路径是VAL扫描，没有收到新增TEST结果。历史TEST已多次暴露的限制继续保留。

| 候选 | real全帧中心正确覆盖@15px | real平均RIoU | sim原协议A-RMSE° | sim平均RIoU | real/sim最长RIoU失败 | 原选权中心约束 |
|---|---:|---:|---:|---:|---:|---|
| epoch16 | 95.20% | 0.7521 | 2.9964 | 0.8268 | 3 / 0 | 未通过 |
| epoch18 | 97.60% | 0.8010 | 4.4161 | 0.8696 | 1 / 1 | 通过，唯一可行 |
| epoch20 | 95.20% | 0.7832 | 1.9767 | 0.8627 | 4 / 0 | 未通过 |
| epoch22 | 94.93% | 0.7811 | 1.9311 | 0.8677 | 2 / 1 | 未通过 |
| epoch24 | 95.20% | 0.7847 | 1.7850 | 0.8759 | 2 / 1 | 未通过 |

五者MCML均<=5；加权中心最优Wmax=0.9928、下界0.9878。epoch24的加权中心仅0.9856，软评分最高不能越过硬约束。保持epoch18，不能因其角度较差改选epoch24，也不扫描其他epoch或放宽门槛。

| 固定VAL比较 | B epoch24（已有逐帧报告） | F-S epoch18（本次终端，缓存待核） |
|---|---:|---:|
| real输出覆盖 | 374/375，99.7333% | 375/375，100% |
| real输出帧中心命中@15px | 360/374，96.2567% | 366/375，97.6000% |
| real全帧中心正确覆盖@15px | 360/375，96.0000% | 366/375，97.6000% |
| real全帧平均RIoU | 0.795505 | 0.8010 |
| real最长无输出 / RIoU失败 | 1 / 4 | 0 / 1 |
| sim输出/条件中心命中/全帧正确覆盖@15px | 三者均512/512，100% | 三者均512/512，100% |
| sim全帧平均RIoU | 0.885695 | 0.8696 |
| sim最长RIoU失败 | 0 | 1 |

F-S ep18转换887框、正式推理max_per_img=1，支持全部887帧有输出的上述推算；其原始缓存对应关系仍待比较脚本核对。MCML按RIoU>=0.5判定成功，不能把sim MCML=1误说成无输出1帧或15px中心失败1帧。real多输出1帧、中心正确多6帧、RIoU约增加0.0055；sim RIoU下降约0.0161。这不是所有指标都差，但**已显示固定`sim_all_frame_riou_not_worse`不满足**；其余16项尚不能凭终端判断。五个候选sim RIoU均低于B，未见可通过重选当前五者解决该指标的证据。

### 41.3 角度与中心评价口径（代码事实、有限推断）

训练在线验证real中心阈值25px、sim10px；正式离线选权两域中心15px。在线`Weighted_R_center`不能与离线Wmax门槛混用。两种中心阈值均为原协议，当前不修改。对外中心命中仍只统计输出帧，并另报输出覆盖及全帧中心正确覆盖。

离线sim原协议A-RMSE在中心>=10px或没有输出时，计90°，不是所有输出框的纯π周期角度RMSE。ep18在线sim中心召回0.9980约为511/512，而离线15px为512/512，在线/离线A-RMSE同为4.4161°。

**推断，待逐帧缓存确认：** 若在线/离线帧对应一致，则有1帧中心在[10,15)px，被计入90°角度惩罚，贡献约81.12%的原协议平方误差。去掉这1个惩罚后，其余511帧角度RMSE由舍入汇总估算约1.9206°。这不是完整512输出帧的纯角度RMSE，被惩罚帧的真实角差未知，不能据4.4161断言F-S纯方向一定退化，也不能据1.9206宣称完整纯方向条件已通过。现有缓存比较会直接计算所需纯角度，无需新推理。

### 41.4 TRAIN收敛、分类与显存（事实及限制）

日志共301行：1元数据、288个TRAIN窗口、12次偶数epoch在线VAL。24epoch各有iter50至600的12个窗口，无重复、倒序、缺epoch，必要loss/grad/positive/memory/lr字段均有限且非零；日志采样不等于每步完整记录。总loss与实际各loss之和最大差2e-5，符合舍入量级。

| 窗口 | 主分类loss中位 | 主KLD中位 | 已加权尺寸loss中位 | 裁剪前梯度范数中位/p90 | 尺寸/KLD比中位 |
|---|---:|---:|---:|---:|---:|
| epoch1 | 0.137915 | 0.583560 | 0.013600 | 44.4774 / 112.4176 | 2.1305% |
| epochs5–16 | 0.054140 | 0.044195 | 0.001735 | 2.9296 / 3.7710 | 3.9399% |
| epochs17–24 | 0.020670 | 0.015535 | 0.000735 | 1.7693 / 1.8999 | 4.6803% |

尺寸项始终参与记录；`shape_positive_count`从O2M的18/卡逐渐转为O2O的2/卡，符合原分配阶段，不能当成检测输出减少。首窗口总loss3263、grad约461049，B已有首窗口也约3263/461046，不能将共同初始化大值归因于新尺寸项。epochs5–24没有任何日志窗口平均grad_norm>10，未见持续梯度爆炸；窗口均值不能证明每一步都未裁剪，也不能推算精确裁剪率。

复用已核验B训练报告相同288个epoch/iter窗口，而非重跑B：epochs17–24 F-S分类中位0.020670比B0.015965高29.47%，KLD中位0.015535比B0.015985低2.82%。后期尺寸项和KLD下降，real覆盖却在ep18之后回退，说明优化目标下降没有保证固定VAL联合几何指标改善。**推断边界：** 分类差异值得记录，但没有real/sim分域梯度或因果干预证据，不能据此认定分类竞争、权重过大/过小或过拟合是根因；标量尺寸/KLD比不是梯度比，不能直接据4.68%调lambda。

全部288条TRAIN的`memory`均3007MiB（约2.94GiB），已记录的累计allocated峰值没有增长，上传记录未出现OOM。它不是nvidia-smi显存，也不是B/F-S隔离额外占用对照；没有最后每步及reserved完整轨迹，不能承诺任意后续任务均无显存峰值。当前结果不支持尺寸项引起逐epoch显存累积的担忧。

### 41.5 下一步：只补已有选中缓存的固定比较（建议，尚未执行）

**当前保留B ep24，冻结F-S ep18，不推荐本轮新增TEST、调损失系数、延长训练或重选权重。** 已有sim重叠退化足以阻止直接替换B；尺寸是否改善、普通real中心保护与纯角度条件仍应完成既定报告，才能准确记录此单次实验的局部收益及代价，不提前给17项失败数量。

若已有比较报告，直接回传；若尚未生成，服务器仅运行现有CPU缓存比较，不重跑TRAIN预检、推理或扫权重。输出已存在时程序拒绝覆盖，保留并读取原报告。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python crane_project/tools/compare_port_size_f_s_val_v1.py \
  --require-reviewed-library \
  --f-sweep work_dirs/crane_symeood_k1_port_day2night_size_f_s_v1/val_sweep_port_v1 \
  --out-json work_dirs/port_size_f_s_v1_val_compare.json
```

回传`port_size_f_s_v1_val_compare.json`及同一F-S VAL目录的`sweep_results.json`；已有正式合同JSON可同时回传。只补逐帧几何、17条件、RIoU交叉核对和身份缺口，不追加完整审计。没有联合收益就将F-S记录为固定设计下未达到目标的实验，继续以B开展第36–40节已经收束的当前帧分量可靠性路线；该路线是质量识别研究，并不自动修正原框几何。不因本次F-S结果自行改动正在准备的可靠性源码或启动新实验。

本轮仅读取、静态核对、复算与归档结果，并更新本文；未修改检测/可靠性源码、启动训练/推理、连接服务器或读取新增TEST。TEST历史多次暴露、VAL用于选权/开发、单seed及历史B对照的结论限制保留；等比例变换、原图还原与独立深度真值缺口均未改变。

## 43. 结构可靠性有限TRAIN预检回传：工程接入通过，方向监督边界需收束（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-43)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 42. F-S VAL缓存比较复核完成，按用户授权准备固定epoch18 TEST（2026-10-02）

### 42.1 输入、身份与代码复核（事实）

用户上传`port_size_f_s_v1_val_compare.json`和服务器比较终端，要求读取、检查代码、分析并给TEST指令。本轮按原字节归档，保留第41节尚缺比较报告的历史状态，以本节关闭该缺口。

| 归档/复核文件 | SHA256 |
|---|---|
| `work_dirs/port_size_f_s_v1_val_compare_server_20261002.json` | `b382caca0d9264344906e3e9393ca5906f2a86e383182c5b51e3fa4f594a3117` |
| `work_dirs/port_size_f_s_v1_val_compare_terminal_server_20261002.txt` | `0690e4c2cfd6223d7b581c5a8f924e460547da2bd10ba55d63106a923dadc85c` |
| `work_dirs/port_size_f_s_v1_val_compare_review_local_20261002.json` | `645eaa82f5e1bf94c4820c46e8536d6e46ae98d18bede14eac5204646eb131de` |

终端打印的条件JSON与报告对应字段完全一致。正式合同SHA、服务器预检SHA及B身份均与第34节一致，`library_matches_reviewed_preflight=true`。服务器检查的B/F-S checkpoint训练配置均`MATCH`，seed0、epoch24/18、累计iter15360/11520一致。F-S依原VAL规则选择epoch18、1/5可行、`constraint_pass`，没有fallback或按比较重选。

F-S冻结身份：checkpoint SHA `ccfe1133b9e0fdb8f79de1b0cb272203d81b87b1a27d45377b3200fbdff47259`；VAL PKL SHA `8bded40f92333a507ffdbe520ac7634447254e76ac919c704d5fb3090b09024e`；选权JSON SHA `821bb9b67c0512697a5b41aeab398f868447aa3e5e16b6c89476fa4ec2ff24bd`；正式配置SHA仍`441aa8ace4621c9eda00a80b764c76c53b82106e73f1987a4bc96c761e8ff758`。

**本地必要复核：** 读取比较入口、条件计算、几何分解、缓存加载、LogSizeLoss及固定TEST/分序列入口；当前41项冻结源码仍匹配。两臂各887帧（real375/sim512），图像ID顺序/唯一性、图像SHA与GT配对一致；B逐帧记录与此前冻结B报告完全相同。用独立标准库公式复算中心、排序长短边误差/signed log ratio、π周期纯角度、惩罚、各组mean/median/p90/RMSE与配对改善数量；用序列/编号缺口规则复算失败区间，再独立组装17项布尔比较，均一致。原比较函数回放亦一致。

RIoU float64交叉检查回放一致：B最大差1.4677e-5，F-S最大差0.000689878；均无>1e-3项及0.5阈值判定变化，`metric_consistency_review_required=false`。没有发现可解释当前失败的分母、配对、门槛方向或RIoU阈值数值错误。原测试已通过且相关源码未变，本轮不重复同版本测试；新增回传数据的数学复核已完成。服务器原权重/PKL/sweep字节仍未上传本机，其SHA属于服务器已核验报告，不能写成本地直接读取原文件。

合同内`optimizer_steps=0`/`formal_training_executed=false`表示CPU合同检查自身没有执行训练；不是说正式F-S未训练或训练失败。该字段仅有阅读口径易混淆，不影响评价。本轮没有自动修改只读审查的源码。

### 42.2 17项条件及实际取舍（事实）

**10项通过、7项失败，`all_conditions_met=false`；F-S不替换B。** 失败为两域长边误差下降（2项）、两域共同输出中心mean不增（2项）、sim全帧RIoU不降（1项）、普通real共同输出中心mean/RMSE保护（2项）。两域短边误差及sim纯角度均改善，不能再概括为尺寸和方向全部退化。

覆盖再次确认：real B输出374/375、输出中心360/374=96.2567%、全帧中心正确360/375=96%；F-S输出375/375、输出中心及全帧正确366/375=97.6%。sim两臂输出覆盖、条件中心命中及全帧正确覆盖均512/512。阈值仍15px，条件中心率仅以输出帧为分母。

| 既定配对/全帧指标 | B ep24 | F-S ep18 |
|---|---:|---:|
| real全帧平均RIoU | 0.795505 | 0.801033 |
| real最长无输出 / RIoU失败 | 1 / 4 | 0 / 1 |
| real共同输出长/短边平均相对误差（n374） | 8.4208% / 8.7580% | 9.2371% / 7.5428% |
| real共同输出中心mean/RMSE，px | 9.5746 / 34.5779 | 11.3402 / 66.1317 |
| 普通real共同输出中心mean/RMSE（n364），px | 4.4375 / 5.2923 | 10.1442 / 65.1515 |
| 普通real共同输出RIoU（n364） | 0.819545 | 0.809662 |
| sim长/短边平均相对误差（n512） | 3.8712% / 4.1388% | 5.6969% / 3.7872% |
| sim中心mean，px | 1.6452 | 1.7547 |
| sim完整输出纯角度RMSE | 2.107151° | 1.920707° |
| sim全帧平均RIoU | 0.885695 | 0.869585 |
| sim最长RIoU失败 | 0 | 1 |

sim长边p90从7.7625%升至10.5489%，长边mean signed log ratio从-0.022673降至-0.056288；短边从-0.031747到-0.029720。支持当前sim长边偏小倾向加重而短边改善，不是等比例两边同步改善。real长边也略退化，不能据尺寸loss下降声称尺寸总体更准。sim共有352/512帧长边误差变差、319/512帧RIoU变差；并非仅那个协议角度惩罚帧影响整体。

### 42.3 新严重错位与角度惩罚（事实、推断边界）

F-S real_seq07零RIoU输出共7帧，其中3帧属于B历史严重组，4帧是新增：

| 新增帧 | F-S中心误差，px |
|---|---:|
| real_seq07_00003 | 769.7530 |
| real_seq07_00186 | 408.5113 |
| real_seq07_00194 | 855.4976 |
| real_seq07_00200 | 212.1999 |

这4帧全部保留在事先固定的普通real组，贡献该组约99.4324%的中心平方误差。该组中心median从3.6827降到3.3401px、p90从7.8297降到7.6044px，同时RMSE从5.2923升到65.1515px。说明多数/典型帧改善与新增极端尾部错误并存；不能删除新增错位、重定义普通组或仅报告median来宣称定位整体改善。虽然旧10帧错位组得到部分修复、总体中心正确数增加，新的极端错误仍违反用户的定位可靠性目标。

第41节角度推断已由逐帧记录确认：`sim_seq10_00211`中心误差10.6138px，小于15但达到原角度门槛10px，实际纯角差1.9819°，被原协议计90°。sim完整输出纯角度RMSE确为1.920707°，相对B约下降8.85%；原协议RMSE4.416079°保留。两者分别回答纯方向精度与含中心惩罚的协议表现，不修改原选权指标或把纯角度替换入选权。

上述结果能确认“改善短边/方向但损害长边及部分定位”，不能证明尺寸项系数过大、分类竞争、背景接管或某一梯度机制就是根因。单seed历史B对照及VAL已用于选权/开发的限制保留。不得由待看的TEST再设计系数、阈值或权重。

### 42.4 用户授权的固定TEST及服务器指令（待执行）

用户本轮明确要求“也去test测试看一下”。按该授权提供一次**原VAL冻结F-S ep18**的TEST，观察既定方案在其他视频序列的结果；不是重新筛选模型的阶段。保持原配置、坐标还原、15px中心门槛及全部评价条件，不扫描TEST epoch、不改F-S系数。TEST已多次暴露，不能称未接触的独立测试集；无论结果怎样，禁止依据它改选epoch或反过来放宽VAL门槛。B仍是目前保留方案。

先确认服务器原VAL选权文件SHA为本节冻结值。以下命令复用已有入口、串行使用物理GPU0；若该卡被占用，只将`--gpu`改为实际空闲物理编号。脚本自身设置CUDA_VISIBLE_DEVICES，无需再套可见卡映射。不连接服务器，也不重新运行B TEST。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
FS_CONFIG=crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1_formal.py
FS_WORK=work_dirs/crane_symeood_k1_port_day2night_size_f_s_v1
FS_SWEEP="$FS_WORK/val_sweep_port_v1"
FS_TEST="$FS_SWEEP/final_test/epoch_18"

# 读取既定VAL选择，只执行该权重的TEST；没有新扫描。
python crane_project/tools/ckpt_sweep.py \
  --config "$FS_CONFIG" --work-dir "$FS_WORK" --sweep-dir "$FS_SWEEP" \
  --final-test-from "$FS_SWEEP/sweep_results.json" \
  --center-thresh 15 --gpu 0

# 仅在上述TEST报告已生成后读取同一缓存，分别报告三序列及中心分母。
if test -f "$FS_TEST/final_test_metrics_v2.json" && test ! -e work_dirs/port_size_f_s_v1_test_subsets.json; then
  python crane_project/tools/audit_port_test_subsets_v1.py \
    --gt-dir crane_project/data/crane_grab_port_day2night_v1/test/annfiles \
    --pred-dir "$FS_TEST/preds/Task1_grab" \
    --out-json work_dirs/port_size_f_s_v1_test_subsets.json
fi
```

TEST入口校验原选权角色/版本/配置SHA及选中checkpoint路径/SHA，`--final-test-from`分支直接返回，不运行VAL扫描。期望1440帧（real_seq03=200、real_seq04=668、sim_seq09=572）。既有分序列脚本提供输出条件中心命中、输出覆盖、全帧中心正确覆盖及无输出/RIoU失败区间；它不是新的完整诊断，也不改预测。它的输出写入没有拒绝覆盖保护，所以**若该F-S分序列报告已存在，直接读取回传，不重复执行第二步**；新的独立文件名与旧B/E-H报告不冲突。TEST入口保留旧预测及不相同的最终报告，身份不匹配时拒绝，不绕过检查。

回传上述`final_test_metrics_v2.json`、`work_dirs/port_size_f_s_v1_test_subsets.json`及TEST末尾终端。保留同目录`preds/results.pkl`、其provenance及`Task1_grab`供必要时复用，不默认再添加全量诊断。当前命令主要评估覆盖/重叠/时序及原协议角度；单独尺寸/纯角度若需解释，应复用既有框，不能用后续结果重写方案。

完成后按B固定TEST旧结果与F-S ep18新结果报告各域/序列的收益及代价，再收束本次F-S实验。后续可靠性研究继续使用已固定B，不能依据这次TEST切换前端；不自动增加新的损失试验或重启已关闭路线。本轮只有本地结果复核、原字节归档和本文更新；没有改检测/可靠性源码、执行训练或TEST、连接服务器。新的TEST仅待用户在服务器执行。

## 44. 固定F-S epoch18 TEST回传：real覆盖与连续性下降，保留B（2026-10-02）

### 44.1 身份、终端与汇总复核（事实）

用户提供`final_test_metrics_v2.json`、`port_size_f_s_v1_test_subsets.json`及TEST终端。本轮只分析该已授权固定实验结果，不重跑TEST、连接服务器或修改检测/可靠性代码；继续集中更新本文。

| 原字节归档/本地复核 | SHA256 |
|---|---|
| `work_dirs/port_size_f_s_v1_final_test_metrics_server_20261002.json` | `3d07da6c126fece22aef3442cffdd37c8ec59ff99cafd5220f881574dca74ecd` |
| `work_dirs/port_size_f_s_v1_test_subsets_server_20261002.json` | `1b29db3e3477c7471032eadb7b43dbc24e9bd1f38a8cb81b0fd7b7840ec1d076` |
| `work_dirs/port_size_f_s_v1_test_terminal_server_20261002.txt` | `44d032401105feaaf60a88b46fe5380e11b08c4f804466aaaa5c133458b1bd0c` |
| `work_dirs/port_size_f_s_v1_test_review_local_20261002.json` | `30650dca4b46ad221abd270d1f6b5455a6da0c681398ea20a574a4c8db32e674` |

主报告为metric_protocol_version2、fixed_test_after_source_val_selection、epoch18、15px、1440帧。配置SHA仍`441aa8ace4621c9eda00a80b764c76c53b82106e73f1987a4bc96c761e8ff758`；checkpoint SHA仍`ccfe1133b9e0fdb8f79de1b0cb272203d81b87b1a27d45377b3200fbdff47259`，均与原VAL冻结身份相同。TEST PKL SHA记录为`29f900ababb2579a4220647d234e8e075fde451058c338e0387c980cd2dc15ca`。TEST标注SHA `e0dbb1bd8aea7209314d8ed60bc44e965550ed606135cc0016e1075d717de13e`与本地1440份标注逐字节集合SHA及此前E-H固定TEST一致。

终端最终逐序列JSON与附件完全一致；终端总体/逐序列指标与两个JSON匹配。独立反算整数中心/RIoU正确数，复核所有覆盖分母、无输出RIoU计零的归一化、区间长度/总数/最大值及合并real按帧加权的中心和RIoU，均一致。总输出1410=170+668+572，与终端转换一致。上传终端未见推理失败，MMCV输出为既有版本提示。

B对照重新读取原固定B ep24终端（第26节同一附件），来源SHA `32d0445c04cc1e730a260b8b3c40efb4d2e98b8fbc51c901ef8f5c3f74b86525`，只读取其中B首段，未将后面的C结果混入。B1431框、sim全输出与固定分母反算real859输出；98.85%舍入覆盖唯一对应858个正确中心。B完整逐序列报告/预测仍缺，不能冒用旧992帧K1缓存或给出B/F-S逐帧交集。本轮没有服务器权重/PKL/TXT原文件，相关SHA属于服务器报告核对，不宣称本地直接重算预测或中心距离分布。

### 44.2 B / F-S 固定TEST总体对照（事实）

| 指标 | B ep24 | F-S ep18 |
|---|---:|---:|
| 总输出帧 | 1431/1440 | 1410/1440 |
| real输出覆盖 | 859/868，98.9631% | 838/868，96.5438% |
| real输出帧中心命中 | 858/859，99.8836% | 832/838，99.2840% |
| real全帧中心正确覆盖 | 858/868，98.8479% | 832/868，95.8525% |
| real无输出数 / 有输出中心不正确数 | 9 / 1 | 30 / 6 |
| real全帧mean RIoU | 0.8169 | 0.8007 |
| real DFR / ACI | 2.5103 / 0.9440 | 2.5737 / 0.9410 |
| real最长RIoU失败 / MCML mean | 4 / 3.5 | 9 / 5.5 |
| real MCML<=5 | 通过 | 未通过 |
| real MRF | 2.29 | 2.85 |
| sim输出/输出条件中心命中/全帧正确覆盖 | 三者均572/572 | 三者均572/572 |
| sim原协议角度RMSE | 1.8823° | 2.0649° |
| sim全帧mean RIoU | 0.8816 | 0.8771 |
| sim DFR / ACI | 2.4734 / 0.9479 | 2.7905 / 0.9427 |
| sim最长RIoU失败 | 0 | 0 |
| 两域TDR_w10 | 各100% | 各100% |

F-S比B少21个输出、正确中心总数少26，全帧中心正确覆盖下降约2.9954个百分点；这是总体计数差，不能说已证明“同样26个B正确帧被F-S丢失”。real重叠下降约0.0162、最长RIoU失败增加5帧；sim角度增加0.1826°（约9.70%）、RIoU下降约0.0045，DFR/ACI也退化。TDR100%不表示全帧正确或连续；其定义为每10帧窗口至少一次RIoU>=0.5，与最长9帧失败并不矛盾。MCML mean是既定分段最长失败的均值，不是全部失败区间长度均值。

### 44.3 各视频序列的输出、中心与连续性（事实）

| 序列 | 输出覆盖 | 输出条件中心命中 | 全帧中心正确覆盖 | 全帧RIoU | 最长无输出 / RIoU失败 |
|---|---:|---:|---:|---:|---:|
| real_seq03 | 170/200，85% | 165/170，97.0588% | 165/200，82.5% | 0.669931 | 9 / 9 |
| real_seq04，夜间 | 668/668，100% | 667/668，99.8503% | 667/668，99.8503% | 0.839888 | 0 / 2 |
| sim_seq09 | 572/572，100% | 572/572，100% | 572/572，100% | 0.877104 | 0 / 0 |

30个无输出全部在seq03。原编号区间为97、107–115、123–124、126、133、141–145、147、168、173、179、182–188，总长30；最长107–115为9帧，另有7帧和5帧连续缺失。该序列RIoU失败共35帧（30无输出＋5有输出失败），有输出RIoU均值0.788154，按全部200帧计入缺失后为0.669931。只报97.06%输出条件中心命中或0.788154会掩盖覆盖代价。

seq04全668帧有输出，只有1个中心错误输出、2帧RIoU失败（205–206）。seq03也有5个中心错误输出，但中心错误帧ID未提供，不能仅因数目等于5个有输出RIoU失败就认定为同一批帧。B没有同序列完整报告，因此不宣称夜间seq04相对B提高，也不精确分解B到F-S的每序列损失。可以确认F-S自身的缺失集中seq03，不能据此在后续训练设计中使用这个TEST困难段调规则。

sim572帧若有1个90°协议惩罚，RMSE至少90/sqrt572=3.7631°。B1.8823°和F-S2.0649°均低于此界，依当前固定实现可排除中心门控/缺失惩罚。因此，F-S此次TEST sim角度退化不能归因于第42节VAL单帧90°惩罚；仍未读取原预测逐帧重算纯角度/宽高。

### 44.4 联合判断、限制及下一步（事实、推断、建议）

**F-S固定实验未达到联合几何目标，保留B ep24。** 原VAL17项失败7项，短边和sim纯角度局部改善伴随长边与中心尾部代价；此次固定TEST没有重复real覆盖/RIoU收益，sim角度也没有重复VAL改善。不同视频序列确实存在表现差异，此次评估有报告价值，但没有提供替换B的证据，也不能仅凭单次结果认定过拟合、系数大小、分类竞争或梯度机制为根因。

当前TEST文件不含长/短边误差、中心距离分布或逐帧原框，不能把TEST的RIoU下降直接归因于VAL观察到的长边低估。保持等比例变换/原图还原不等于深度已改善；独立real深度准确性仍未验证。B本身仍有9个无输出及10个全帧中心不正确，保留B不表示已经实现所有real全帧正确覆盖。TEST历史多次暴露、VAL用于选权/开发、单seed和视频帧相关性继续披露，不声称显著/稳定收益或未接触测试集上的独立确认。

下一步收束为：

1. 将D/E-H/F-S分别保留为各自固定设计的取舍/未达到目标实验，不据此次TEST换F-S epoch24、增加系数、改阈值/数据划分或继续扫描。当前不需要新的TEST或完整几何审计；现有证据足以完成“保留B”的判断。
2. 返回已经预登记的冻结B当前帧分量可靠性路线。结构有限TRAIN接入回传已确认工程通过（同文“结构可靠性有限TRAIN预检回传”节）；下一阶段应按TRAIN证据先锁定原始轴线方向监督资格、正式训练/保存/验证协议及普通ROI/结构分支公平预算，再按授权实施固定对照。不能从此次TEST选择特征、阈值、前端或结构设计。
3. 可靠性分量评分检验的是识别可接受中心/尺寸/方向的能力，不直接改原框，也不能恢复无输出。保持原B中心利用及输出，尺寸/方向低质量不能自动转为整帧拒绝；优先用TRAIN/VAL固定条件验证相对score/静态几何及普通ROI的新增价值。

若以后确需补充完整F-S结果表，可只读已保存的B/F-S TEST框补尺寸/定位统计，属于报告完善而非调参；目前不要求这一步，也不重跑B或F-S推理。最新工作状态是F-S固定TRAIN/VAL/TEST已经回传并完成汇总复核；可靠性正式拟合尚不能由本次TEST自动放行。本轮仅归档、复算和更新本文，未修改源码、运行训练/推理或连接服务器。

## 45. 继续几何精度优化：问题本质、原因边界与定向文献检索（2026-10-02）

### 45.1 最新授权与复用范围（事实）

用户明确本对话继续几何优化，另一个对话进行可靠性分析；本轮任务为总结问题、本质与原因并搜索论文。第44.4节转向可靠性的建议不再作为本对话工作方向；其他对话的结构/质量分支及记录保持。本轮仅复用已有TRAIN/VAL、固定实验结果和必要静态接口，做定向文献核对及本文更新，没有修改检测/可靠性源码、连接服务器、训练或推理。

重点复读第7/16/17/24/28/30/31/41（F-S）/42/44节，并读取`work_dirs/port_size_f_s_v1_val_compare_server_20261002.json`、`work_dirs/port_shape_e_h_v1_val_compare_server_20261001.json`和已复核的F-S TEST本地报告。现有VAL保存框已足以说明几何取舍；不重做完整审计，也不要求补新的TEST框来设计方法。B ep24、D ep22、E-H ep22和F-S ep18均保持各自原VAL选权身份。

### 45.2 主要问题：普通几何误差与严重尾部必须同时报告（事实）

| 已确认问题 | 现有证据及其含义 |
|---|---|
| 普通real定位仍有可优化误差 | B既定普通共同输出组n364，中心mean/RMSE为4.4375/5.2923px；并非全部定位问题都由历史10个严重错位帧造成。该组长/短边平均相对误差为7.8524%/7.9396%。 |
| 两域存在尺寸低估倾向，不能将尺寸问题简化为长宽比 | B普通real长/短边mean signed log ratio为-0.055679/-0.067992，sim为-0.022673/-0.031747。负均值表示整体低估倾向，不表示每个框都偏小，也不自动给出统一放大系数。sim B长/短边相对误差为3.8712%/4.1388%。 |
| 尺寸是当前重叠误差的重要来源 | 已有B保存框只替换一个分量为GT的RIoU反事实：普通real替换尺寸/中心/角度的mean增益为0.074417/0.027273/0.004357；sim为0.030196/0.020994/0.011274。尺寸替换收益最大，支持优先研究尺寸与局部定位；这些非线性反事实不能相加当因果贡献比例，GT替换也不证明模型能够学到对应增益。 |
| 纯角度可以改善，但没有形成联合收益 | E-H在VAL sim纯角度RMSE从B的2.107151°降至1.706917°，但长/短边误差升至5.2978%/5.1285%，RIoU由0.885695降至0.873045。F-S角度降至1.920707°、短边降至3.7872%，长边升至5.6969%、RIoU降至0.869585。B相对EOOD同增强的主要sim差距是中心/尺寸，不能把纯角度当唯一瓶颈。 |
| 典型框改善与新增严重错位可以并存 | F-S普通real中心median由3.6827降至3.3401px，p90也略降，但新增4个零RIoU错位使同组RMSE由5.2923升至65.1515px；这4帧贡献约99.4324%中心平方误差。必须保留其总体代价，不能移除新增错位或改分组。数百像素错位与几像素定位误差属于不同误差尺度，原框不能证明是选错对象还是回归大偏移。 |
| 覆盖尚未达到用户理想目标 | B real VAL输出374/375=99.7333%，输出中心命中360/374=96.2567%，全帧中心正确360/375=96%。sim三者均512/512。B保留不意味着real已经全帧正确。 |

D改善部分real中心与覆盖但未稳定改善尺寸；E-H/F-S有分量收益和代价。三次固定实验没有达到联合目标，足以反对直接宣布“额外监督就解决问题”，但不能排除所有其他损失设计，也不能证明三个实验具有同一根因。

固定TEST仅保留历史报告含义：F-S real输出覆盖96.5438%、输出中心命中99.2840%、全帧正确95.8525%，B分别为98.9631%、99.8836%、98.8479%；sim原协议角度B/F-S为1.8823°/2.0649°。它说明该固定方案未在其他序列重复VAL收益，不据TEST困难序列、尺寸猜测或角度结果选择下一个方法、阈值或epoch。当前TEST已多次暴露，VAL也用于开发/选权，单seed及视频帧相关性限制保留。

### 45.3 问题本质及原因分层（事实、机制推断、未知）

**当前可以确认的本质：现有监督与优化结果之间存在联合几何取舍；精度缺口同时涉及普通框偏差与少量严重错误，不能靠一个平均损失或一个平均指标描述。** 不是已经查明某个loss缺失或某行代码导致全部退化。

| 原因/解释 | 当前证据等级 | 能说明什么、不能说明什么 |
|---|---|---|
| 联合KLD本身的参数耦合 | 数学事实＋既有实际节点检查 | 形状项联合处理尺寸/方向，反向KLD中心项受预测协方差影响；独立F-S尺寸项的正确边长导数也不保证主KLD＋F-S总导数或共享参数更新纠偏。耦合也可能有益，不能直接判定其造成总体偏小。 |
| 修改回归会影响共享表示和预测相关训练流程 | 代码事实；具体失败因果未确定 | 当前B分类/回归预测参数各自独立，但读同一FPN/backbone特征；SymNFL几何权重及分配使用detach预测，虽不沿该路径直接反传，后续预测变化仍会改变权重/分配。E-H/F-S不直接监督中心/分类仍可改变输出与中心。没有完整历史反事实证明哪个路径主导。 |
| 几何特征的空间/方向/尺寸对齐不足，或主头细化能力有限 | 静态结构事实＋待验证假设 | 复核`SymEOODHead.forward_single`及`RotatedRetinaHead._init_layers`：主头是直接3×3分类卷积及五参数回归卷积；配置`stacked_convs=4`不等于该实际路径执行四层塔。FPN步长8/16/32/64/128；主头没有根据当前预测OBB重采样。此事实已在第16节记录，不是新发现的bug。它支持检验局部几何特征/细化，尚无证据确认特征错位是当前主要根因；步长也不是定位精度的绝对下限。 |
| 优化目标、正样本监督与最终输出评价并非同一对象 | 代码/协议事实；贡献大小待验证 | 正样本KLD和新增项下降不等于固定score阈值/top1最终框的每条边误差下降。原VAL选择偏重覆盖、时序及协议角度，不直接优化全部边长/中心/RIoU指标；原选权可行不等于通过联合目标。F-S扫描各epoch的sim RIoU均低于B，因此不能只归咎于选了epoch18。保持历史选权规则，不用新报告改选。 |
| 后期梯度爆炸、裁剪压掉新增项或数值/导出故障 | 既有证据不支持作为首要解释 | B/E-H/F-S初始化大梯度为共享现象，后期记录和有限探针未见持续大梯度/clip主导；F-S记录显存恒为3007MiB。实际loss接入、导数、配对、分母、RIoU数值及身份已核验。日志是窗口统计、探针只覆盖有限TRAIN，不能排除所有未采样异常。 |
| 缩小增强、角度表达边界、标注与跨序列差异 | 部分事实＋根因未知 | 有限TRAIN clean下B/EOOD几何差距已存在，两者half后均下降，不足以归因增强B。表示等价已处理，现有GT角度未集中在le90边界，不优先归因角度周期。real短边由轴线长度L/k构造，约束是标注参考框而非整个抓斗外轮廓；该定义可能影响图像监督方式，但未量化标注误差或证明是低估根因。单seed和不同选择epoch不能区分优化取舍、方差与序列泛化原因。 |

因此，下一项优先检验的假设是：**在不扰动B原检测决策的条件下，预测框附近更贴合参考结构的局部图像特征，能否支持尺寸/方向及有限定位的准确细化。** 这是优先级判断，尚不是已证实原因，也不自动批准新增模块训练。

### 45.4 定向论文检索与问题—机制—证据映射（已检索事实）

检索日期2026-10-02。关键词覆盖oriented feature alignment、rotated RoI、coarse-to-fine refinement、scale-orientation coupling、decoupled regression及gradient calibration。只依赖作者预印本、官方会议/期刊原文、作者代码和机构页；是针对性检索，不是系统综述或穷尽原创性检索。直接打开部分CVF/MDPI/作者PDF被拒绝，以下明确核读层级，不将可读摘要/索引段落声称为阅读全文。

| 原论文、身份与核读层级 | 原论文支持的机制 | 对本项目的借鉴与限制 |
|---|---|---|
| Han等，**S²A-Net / Align Deep Features for Oriented Object Detection**，作者页面标注TGRS2021；[作者全文v3](https://arxiv.org/html/2008.09397v3)，III-B～D、式1～6；[作者代码](https://github.com/csuhan/s2anet) | AlignConv按框中心、边长、方向计算采样位置；方向敏感特征供回归，方向不变特征供分类 | 最直接对应几何特征对齐假设。优先借鉴确定性OBB引导采样，避免同时引入新分配/ARF/分类分支；完整S²A-Net含新检测头及NMS，不直接替换当前K1流程。论文不能证明本项目错位根因。 |
| Yang等，**R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object**，AAAI2021，DOI10.1609/aaai.v35i4.16426；[官方PDF](https://cdn.aaai.org/ojs/16426/16426-13-19920-1-2-20210518.pdf)，Refined Rotation RetinaNet、算法1、FRM及消融 | 框细化后重新对齐特征；中心及四角作双线性插值，粗到细回归 | 支持“重新取特征再细化”，避免只复用同一个未对齐特征向量。原模型为端到端多阶段，含框过滤/重新分类/损失与匹配变化；冻结B＋单输出细化是本项目候选适配，不是原论文方案复现。 |
| Ding等，**Learning RoI Transformer for Oriented Object Detection in Aerial Images**，CVPR2019，DOI10.1109/CVPR.2019.00296；会议身份由官方检索核对，[可读作者预印本](https://arxiv.org/html/1812.00155)第3.1～3.3节、式3/4 | 用OBB决定旋转RoI采样格，再回归相对偏移 | 支持围绕B已输出框提取局部几何特征。预印本标题为“for Detecting Oriented Objects”，与会议题名区分；不照搬其HRoI学习、匹配或NMS。目标参考框是上梁/轴线派生框，应核对采样是否取得该结构而非错误外轮廓。 |
| Xie等，**Oriented R-CNN for Object Detection**，ICCV2021；官方身份核对，[作者全文](https://arxiv.org/html/2108.05699)第3.2/3.2.1节 | oriented proposal→rotated RoIAlign→框偏移细化 | 提供可实现的旋转RoI细化参考。完整论文改RPN、分类及NMS；本项目只借鉴几何特征提取，不能据其AP/FPS保证当前几何收益或资源成本。 |
| Liu等，**SFRADNet: Object Detection Network with Angle Fine-Tuning Under Feature Matching**，Remote Sensing2025，17(9):1622，DOI10.3390/rs17091622；[期刊原文](https://www.mdpi.com/2072-4292/17/9/1622)，通过发布页索引核读3.2.1/3.2.2节、式14及4.5讨论 | 尺度特征与方向特征相关；DGBox区域/方向引导采样 | 新近且针对尺寸—方向取舍的参考。仅借鉴形状与方向共同引导取样，不复制新backbone/多角卷积或tan角编码。论文亦承认近方形与推理速度限制；不能将其整套AP归为单个模块、称其解决本项目根因。 |
| Yu等，**SODE-Net: A Slender Rotating Object Detection Network Based on Spatial Orthogonality and Decoupled Encoding**，Remote Sensing2025，17(17):3042，DOI10.3390/rs17173042；[作者机构摘要](https://researchonline.jcu.edu.au/88998/)、[原发布页](https://www.mdpi.com/2072-4292/17/17/3042)索引图3/表3，未读完整推导 | 分类、位置/尺寸和角度使用不同特征分支 | 支持研究特征任务解耦，而不仅新增一个数值分项。表3中已分cls/reg基础上再分位置/角度的mAP为79.76→79.83，不能夸大为大幅解决冲突，更不证明几何分量或本项目覆盖改善。暂不同时改backbone/角编码。 |
| Ming等，**Gradient Calibration Loss for Fast and Accurate Oriented Bounding Box Regression**，IEEE TGRS2024，DOI10.1109/TGRS.2024.3367294；[作者机构页](https://gaim.ugent.be/publication/10440108/)、[作者代码](https://github.com/ming71/GCL)，摘要/身份核读，作者PDF直接打开失败 | 分析并校准rotated IoU的角度响应及尺度敏感性 | 借鉴“核对实际优化响应而非只看loss”。本文主要分析rotated IoU，不能外推其结论到当前SymKLD/H；负相关也不等于导数符号错误。暂不把换GCL作为下一项默认实验。 |

原KLD文献[NeurIPS2021原文](https://proceedings.neurips.cc/paper/2021/file/98f13708210194c475687be6106a3b84-Paper.pdf)与[附录](https://proceedings.neurips.cc/paper_files/paper/2021/file/98f13708210194c475687be6106a3b84-Supplemental.pdf)支持参数耦合及尺度不变性；支持的是联合梯度建模，而非“本项目必须彻底去掉耦合”。第18/31节的ProbIoU、RIL、EIoU、Constraint Loss和RSFPN证据保留，不重复以论文存在为理由再增加同类补偿项。

本次没有找到直接验证“冻结当前港口B＋某一细化模块”能够同时满足既定17项条件的论文；上述工作的遥感AP、细长目标与数据量不能代替本项目的逐帧几何/覆盖对照。论文可信来源核对已完成到表中注明的层级，收益、根因和原创性均未确认。

### 45.5 下一步优先方向与有限验证边界（建议，尚未固定设计）

优先考虑**冻结B后的当前帧局部几何细化**，借鉴S²A-Net、R3Det、RoI Transformer的框引导采样思想；目的是优化实际框参数，可靠性对话中的分量评分保持独立。与D/E-H/F-S不同，此方向增加局部几何信息，并可隔离新增回归对B共享表示及分类/分配的扰动；这是候选工程安排，不是原论文已证明的冻结策略。

1. 保留B ep24、原分类分数/score阈值/top1输出决策；完整冻结backbone/FPN/主头并保持eval、BN状态固定。只以同帧预测框和图像特征训练小型残差细化头，不读取GT在线决策、不引入DINO、教师、候选排序、质量拒绝、跟踪或历史帧。已有`simple_test_from_features`及旋转RoI组件提供静态接入基础，未做该候选运行验证。
2. 首轮可固定B中心，只细化排序边长及周期方向，以隔离尺寸/方向是否确有可学信号。这样在不新增裁剪/过滤、框保持合法且还原一致的条件下，输出帧和中心三项覆盖应与B相同；RIoU失败连续性、DFR/ACI及尺寸/方向仍可能恶化，必须评价。它不能改善中心距离、恢复无输出或修复选错对象，不能声称已解决完整用户目标。若后续需要中心细化，应另明确受限中心残差与定位尾部门槛，不能在本次“固定中心”实验内悄悄放开。
3. 合适的机制对照为B、冻结B＋未按OBB旋转对齐的局部特征细化、冻结B＋按OBB旋转对齐的同预算细化。后两臂统一头参数/样本/损失/训练预算/采样数量与分辨率；只改变几何对齐取样。中心、上下文范围、宽高等价和采样坐标必须有事先定义。对照区分额外细化能力与对齐特征的增量，而不是只和B比较后就认定对齐有效。两臂不在本轮直接进入正式训练。
4. 必要缺口仅为有限TRAIN条件下的可学性与接入核对：当前图像局部采样是否包含标注上梁/轴线结构、实际尺度下空间信息是否足够、相对B真实框残差是否可以被该小头预测、等价OBB是否产生一致几何结果。现有有限TRAIN框可复用；需要特征时由后续获授权的本地代码交付服务器运行，不重做全量审计。若冻结B特征不足，先报告失败，不能自动加高分辨率backbone或放开B训练。
5. 保持等比例输入增强；旋转RoI特征归一化不改原图像/GT长宽比，所有残差通过明确的模型输入坐标→原图坐标链路还原。继续使用原图内参和既有深度估计约束；尺寸角度改进不等于独立深度精度改善。real GT短边L/k定义不改，不硬套统一物理长宽比。
6. 只用TRAIN/VAL确定设计和选权规则，预先记录范围、公式、系数及覆盖/两边尺寸/纯角度/RIoU/定位尾部/时序门槛，保留原实验全部历史规则。冻结模块后的新选权协议如需变化，应在训练前明确理由与独立合同，不能静默重写B/D/E-H/F-S身份。当前不安排新TEST；不因本轮文献或已有TEST结果自动放行训练。

完整用户目标还包括提高中心精度。本节第2点的固定中心候选仅作为先行机制对照；五参数受限细化需要另明确普通real中心mean/RMSE及新增严重错位保护，再按授权实施。本轮完成的是问题归纳、必要静态核对与定向原文检索，尚未冻结下一项正式模型或训练合同。

## 46. 固定TRAIN方向监督资格与真实B错误支持统计入口（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-46)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 47. 真实TRAIN支持回传复核与正式分支入口判断（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-47)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 48. G v1冻结B局部尺寸/方向细化：固定设计与有限TRAIN入口（2026-10-02）

### 48.1 授权、问题与证据边界（事实与待验证假设）

用户授权按第45.5节建议推进。本轮实施“冻结B＋局部几何细化”的设计、有限TRAIN检查代码、CPU验证与服务器指令；没有连接服务器、启动真实数据拟合或正式训练、运行VAL/TEST。可靠性对话的第46～47节及相关代码保持独立，未修改其目标、资格或训练入口。原B/D/E-H/F-S模型、配置、原选权与历史结果不变；B仍为ep24。

**待验证问题：** 冻结B的当前帧P3局部特征，能否支持尺寸/方向纠偏；按预测OBB旋转对齐取样，是否比同容量普通局部采样有增量。借鉴第45.4节S²A-Net、R3Det、RoI Transformer的框引导采样机制；本实现是受限适配，不是这些整套方法的复现或已证明的新损失。

G v1先固定中心，只预测长边、短边、pi周期方向残差。它隔离新增监督对B backbone/FPN、原分类/回归及分配的扰动；细化头内部仍共享表示，尺寸—方向取舍仍可能发生。这个对照不能确认D/E-H/F-S失败的根因，两个细化臂若都改善也不能单独归因于新增图像信息而排除几何描述符/容量的作用。

**目标局限：** B real VAL全帧中心正确覆盖仍为360/375，不是100%。固定中心可保留B中心距离/输出覆盖，不能恢复漏检、改善定位距离或修复已有严重错位；RIoU失败连续性、尺寸/角度时序及深度仍可能恶化。未来受限中心细化需要单独设计和授权，不能在本检查中放开。

### 48.2 固定模型与对照（固定设计）

| 项目 | G v1合同 |
|---|---|
| 检测前端 | 原B ep24，严格config/checkpoint SHA及既有metadata核验；所有模块eval、参数requires_grad=False、grad=None；前后参数/缓冲SHA不变。 |
| 对照 | B零残差基线；ordinary普通局部采样；aligned按当前预测OBB旋转采样。后两臂同头、同初始化、同样本/GT、同loss/优化器/批次/预算。 |
| 采样 | 同帧冻结P3，stride8、256通道；9×9双线性点格；排序L/S各取1.5倍上下文，任一边至少2个P3单元即16输入像素。ordinary角度0，aligned角度为canonical B方向；仅旋转变化，无学习偏移。无效padding源单元先mask，另输入有效支持mask。 |
| 描述符 | 两臂共同使用模型输入坐标的log(L/8)、log(S/8)、sin(2theta)、cos(2theta)，不输入GT、图名、域、序列或资格。 |
| 细化头 | 257→32的1×1卷积、32→32的3×3卷积、flatten＋四维描述符→64→3；无BN/dropout，183,907参数。最后层权重/bias初始化为0，输出初始六列与B逐位相同。 |
| 保留项 | 原B top1、score阈值.05、分数、中心、输出数/输出帧；不重排序、追加候选、重新分类、增加质量拒绝、裁剪或跟踪，不接DINO/教师。 |

令小头输出为z，固定：

\[
 d_L=\log(1.25)\tanh z_L,\quad d_S=\log(1.25)\tanh z_S,
 \quad d_\theta=(\pi/18)\tanh z_\theta;
 \qquad L'=L e^{d_L},\ S'=S e^{d_S}.
\]

原图长/短边倍率分别限制在[0.8,1.25]，方向残差±10°。若L'<S'，两边投影到几何均值，方向保持投影前的canonical长方向；尤其原B raw w<h在投影成正方形后须显式保留此方向，避免排序tie产生伪90°跳变。通常保留raw w/h表示，square tie允许等价方向表示；不以预测近方形为理由移除角度失败。报告投影和≥95%残差上限的饱和次数。

**原图坐标loss：** 三个归一化误差为log(L'/Lgt)/log(1.25)、log(S'/Sgt)/log(1.25)、wrap_pi(theta'−theta_gt)/(pi/18)；分别SmoothL1(beta=.1)，三个分量等权mean，再按batch mean。仅更新细化头；没有新增原KLD分项或B分类/中心loss。宽高交换和pi周期统一按canonical长短边计算。

范围依据只来自原64张TRAIN：两视图两域的边长最大绝对log误差约.15585，小于.22314边长上限；real已有角度尾部超过10°（half最大约15.09°），因此并非所有方向目标均可完全到达。不据尾部扩大界限、不剔除不可达目标；固定上限是首版工程限制，未证明最优。

### 48.3 有限TRAIN预算、数据与坐标（固定设计）

复用原`port_train_val_geometry_v1.json`的SymEOOD TRAIN clean/half各64条现有证据，只提取TRAIN构成项目内fixture。实际固定64张（real32、sim32，覆盖六个TRAIN序列）；每域原固定样本顺序的每第4张为probe，其余fit：fit各24张、probe各8张。同图两个视图始终同角色，图级不交叉；共96 fit视图、32 probe视图。**这是TRAIN诊断probe，B已训练过全部图像，同视频相关性仍在，不能称独立验证或证明序列泛化。**

- 两个视图是原审计的确定性未翻转1024等比例resize/pad，以及其固定0.5等比例缩小。只复用VAL管线定义，不实例化VAL/TEST数据集或读取其图片、标注/预测；没有声称本轮覆盖完整随机B增强/翻转训练链。
- 每视图仅真实B单输出用于拟合；TRAIN离线中心距离<15px才配对监督。至少每域fit8/probe4个有效视图（两尺度合计），不足则停止。没有GT构造框、教师、合成扰动或界限可达性筛选。**在线不根据资格拒绝框：** 包括中心不正确的输出也保留并细化，missing保持空，全部固定视图进入评价分母。
- seed1703；两臂完全相同初始化、预生成批次；batch8，每批4real＋4sim，从fit有效视图有放回抽样。每臂固定200次Adam更新，lr=.001、weight_decay=0、clip_norm10；合计400次新增头更新，B更新0。仅初始与最后step200比较，不选最佳step，不导出拟合权重。
- 复用原TRAIN报告的原图数值GT参考，两臂/视图一致。当前TRAIN图片/标注须精确SHA一致；实际标注解析canonical GT须与参考allclose(atol1e-3,rtol1e-5)，不符则停。最终逐帧报告保存参考GT、实际解析GT、绝对差异及B/细化框，记录OpenCV/torch等版本，不从指纹推断数值完全相同。与第47节可靠性原输入补缺分开，不重新解释其成绩。
- 原B raw模型框先按其w/sx、h/sy约定还原，再排序长短边；不在还原前交换宽高。GT保持原RResize的sqrt(sx*sy)尺寸约定，PortIsotropicShrink仍为单一仿射比例。运行时比对原B native rescale=True与复用坐标函数；历史B TRAIN框重放容差预定为中心/边长.1px、角度.001rad、score.001。
- 旋转9×9是特征采样格，未改变原图或GT长宽比。原图内参与既有深度估计约束不动；改变尺寸/方向仍会改变深度输入，不能据此保证深度精度。

### 48.4 实现、内存、输出与本地复核（事实，CUDA待验证）

新增六个文件，均在项目对应目录：

1. `crane_project/utils/port_geometry_refine_g_v1.py`：两个确定性采样器、同构细化头、原框保留/有界残差及独立loss。
2. `crane_project/tools/preflight_port_geometry_g_v1.py`：固定TRAIN源/数据检查、冻结B特征采样、完整缓存复用、两臂短拟合与全部视图比较。
3. `crane_project/tools/port_geometry_g_v1_train_samples.json`：TRAIN-only fixture、图级fit/probe、原数值GT与旧B两视图框。
4. `crane_project/tools/port_geometry_g_v1_protocol.json`：预定设计、预算、容差与人工审查边界。
5. `crane_project/tools/port_geometry_g_v1_sources.json`：52项来源绑定。原44项本地SHA均保持；运行时保留其43项代码/元数据绑定，排除历史VAL保存框JSON，新增G及关键坐标/采样依赖。原manifest本身仍绑定，不修改旧合同。
6. `tests/test_port_geometry_g_v1.py`：必要CPU数学、梯度、坐标与运行入口回归检查，不是服务器运行依赖。

图像batch1、no_grad冻结B；每张仅缓存两种采样的CPU ROI与mask，不保留整张FPN或检测计算图。128视图全输出时两臂ROI/mask张量约**20.329MiB CPU**，硬上限32MiB；该上限不是进程总RAM承诺。特征采集结束验证B状态并释放B/清CUDA allocator，再进行小头batch8更新。分别记录B前向（含必要native/model输出核对）、局部采样、两臂前向/反向/更新的allocated/reserved峰值；**CUDA实际峰值尚未运行验证，不能保证与历史训练完全一样或绝不增加。**

缓存仅完整成功采集后写`cache_manifest.json`及CPU`local_roi.pt`，绑定源/数据/B、128视图顺序与文件SHA。复用须完整匹配，另验证CPU float32/无图、GT、资格及ROI形状。已有输出或缓存拒绝覆盖；中断缓存不冒充完成。每序列首个fit样本两尺度共12张局部PNG，显示GT参考框、B与两种采样点；这只是结构支持诊断图，不是native轴线标注，也未验证所有像素端点可见。

本地环境torch1.8.0.post3/Python3.8：**24项CPU测试通过**，覆盖零残差六列逐位不变且方向梯度不为零、step2有效stem梯度、GT/输入/检测器不反传、三分量导数符号与中心差分、宽高交换/pi等价、两种采样唯一旋转区别、stride坐标/padding、原w/h还原顺序、正方形投影方向、缺帧分母、同初始化小型入口及非TRAIN路径/来源变更拒绝。最终Python3.8语法检查通过；源绑定和64样本/2558 TRAIN标注身份静态检查通过，B/head真实数据更新0。

最终静态报告：`work_dirs/port_geometry_g_v1_static_check_local_20261002_v4.json`，状态`STATIC_CHECK_COMPLETE_NO_GPU_NO_UPDATES`，SHA `a4a0ed70d705a1fa4e77d077b601f5e34f28f961bc0d18faad33f44b3b41765d`；当前source manifest SHA `b92c8ea792b3066c5543c55163d47c7b1fa1a9634dbab8e393093fe9c61b4063`。开发期v1在manifest尚未生成时失败，v2/v3为后续数值留存/版本记录改动前的检查；均保留，最终交付以v4和当前合同为准。没有本地B权重/CUDA，真实特征、拟合收益、12张实际采样图和显存峰值均待服务器检查，不包装为已完成实验。

### 48.5 服务器运行与回传（已准备，未执行）

上传上述前五个运行文件到服务器项目同名位置；第六个测试文件可一并保留。既有冻结B及历史依赖应已在项目，不需要上传Downloads原几何报告或可靠性轴线报告，不压缩提交包。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_v1.py \
  --check-only \
  --out-json work_dirs/port_geometry_g_v1_static_check.json
```

先确认`STATIC_CHECK_COMPLETE_NO_GPU_NO_UPDATES`，再使用空闲物理卡（示例3，逻辑编号0）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_v1.py \
  --gpu 0 \
  --out-json work_dirs/port_geometry_g_v1_train_preflight.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache
```

正常完成状态`TRAIN_SHORT_FIT_COMPLETE_REVIEW_REQUIRED`，`detector_updates=0`、`head_updates_total=400`、无拟合权重导出。若失败，回传FAILED报告和具体错误，不删保护或改源SHA来绕过。已经完整采集而需复算/重跑同版短拟合时，另取输出名并显式复用，避免重复B特征采集：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_v1.py \
  --gpu 0 --reuse-cache \
  --out-json work_dirs/port_geometry_g_v1_train_preflight_reuse.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache
```

回传主JSON、同名`.progress.jsonl`/`.artifacts.json`、cache的`cache_manifest.json`及12张`previews/`PNG。CPU`local_roi.pt`留服务器，必要时才补；不依赖新增TEST或正式权重。

### 48.6 下一阶段判断（待验证，不自动放行正式训练）

机械要求是冻结B参数/缓冲不变、初始精确B、中心/分数/输出数不变、有效有限三任务/后续stem梯度、合法有界框、来源及坐标一致；零最后层使首步stem梯度为零是预期，第二步必须有效。每步检查loss、梯度和更新参数有限；loss/裁剪/stem日志保留step1、2及每25步快照，不把快照当每步完整轨迹。裁剪快照、最终饱和/投影计数和实际采样图必须审查。

先看每域每尺度fit loss是否确有下降，再看图级TRAIN probe的长边、短边、纯角度RMSE/p90、RIoU及尾部是否联合改善，并比较aligned对ordinary的增量。报告原三口径：输出帧中心命中、输出覆盖、全帧中心正确覆盖；不以仅输出条件指标隐藏无输出或不配对框。稀疏TRAIN样本不报告伪视频连续性/DFR/ACI，也不只挑改善的域、尺度或分量。

完成状态仅表示检查跑完，loss下降不等于局部特征有效或VAL改善。若只有fit改善、probe联合退化，或aligned不优于ordinary，应先报告容量/过拟合/采样等不确定性，不继续堆loss或自动放开B。若工程及可学性证据支持，再另锁定正式两臂预算、保存/重载与G头VAL选权合同；原17项联合门槛保留，不重选B或放宽既有条件。当前VAL已用于开发/选权、TEST已多次暴露；本轮不用TEST选方法、界限、预算或权重，未安排新TEST。

## R1. 可靠性三臂正式训练、保存恢复与固定末轮VAL入口（2026-10-02）

本节可靠性记录已完整迁移至[独立可靠性交接](reliability_handoff_20261003.md#legacy-r1)；原编号、正文、命令、结果与证据身份保留，后续可靠性内容在新文档更新。

## 50. G v1服务器短拟合完成但报告保存失败：修复与旧缓存恢复（2026-10-02）

### 50.1 终端事实、问题定位与结论边界

用户回传终端并询问为何只有一张显卡、是否为完整训练，授权修复。原始终端保存在`work_dirs/port_geometry_g_v1_report_fix_local_20261002/terminal.txt`；未连接服务器，也未启动新的CUDA/正式训练。

**终端事实：** 冻结B ep24加载成功，128视图采集完成，CPU ROI张量缓存20.3291015625MiB；ordinary/aligned均打印到step200。ordinary step1/200的当批loss为.1409845203/.0773882940，aligned为.1409845203/.0787623450。最后在`main -> write_new -> json.dump`报`TypeError: Object of type int64 is not JSON serializable`，程序未正常保存完整主报告。MMCV弃用与meshgrid警告不是这次失败原因；终端未显示OOM或数值失败。20.329MiB是CPU ROI张量体积，不能当作CUDA显存峰值。

**本地已复现根因：** `support_report`原来用NumPy caps做比较，生成`numpy.bool_`，Python `sum`累计后得到`numpy.int64`；旧通用`write_new`不接受此类型。其统计阈值和比较本身未错误，失败发生在报告数值类型。原最终主报告写出又位于运行异常保护之外，且旧writer边序列化边写目标，因而可能留下截断文件，其前部即使出现COMPLETE也不能作为完整完成证据。这个缺陷来自提供的报告代码，已作局部修复。

**结论边界：** 两臂末批loss低于首批是终端观察，不是同一固定样本平均loss比较，也不证明TRAIN probe联合改善；不能凭step200的ordinary/aligned微小差值选臂。尚未收到完整逐帧几何、实际采样预览和CUDA峰值，仍不判断G改善或正式训练放行。

### 50.2 单GPU与训练范围（原固定设计未改变）

原命令`CUDA_VISIBLE_DEVICES=3 --gpu 0`只暴露物理GPU3，并在单进程中使用其逻辑编号0；脚本原本也没有DDP或跨卡任务调度。这符合第48节的有限TRAIN检查：64图、128视图、两臂各200次小头更新、B更新0；并非覆盖全TRAIN的正式epoch训练，也不保存拟合头权重。**GPU数量和是否完整训练是两个不同问题；正式训练也可以单卡执行。**

本次不把原小样本检查自动扩成正式训练、不调整步数或补权重导出。新增显式终端说明及`formal_training=false`、`single_process=true`、`gpu_devices_used`字段，误用多进程WORLD_SIZE>1则停止，避免多个进程重复跑同一检查。正式两臂仍需先依据完整短拟合证据锁定全TRAIN预算、checkpoint保存/重载与VAL选权合同，并实现正式入口；不能直接拿原检测器`dist_train.sh`来训练未接入该入口的G头。

### 50.3 本次修复、来源及缓存兼容（事实）

只修改G运行脚本、G来源manifest、G回归测试及本文；可靠性对话正在修改的其他文件未操作，原检测器/配置和G数学模块均未改动。

1. `target_outside_bounds_by_component`明确产生原生Python整数；G本地JSON转换器把NumPy标量/数组转成数值、布尔与列表。未知对象及NaN/Inf仍拒绝，未用字符串转换或默认填零隐藏错误。
2. 先在内存完整序列化，再同目录临时文件写入/flush/fsync，以原子硬链接发布目标；目标已存在则拒绝覆盖，只清理自己新建的临时文件。最终写报告纳入异常保护，必要时留下原生类型的最小FAILED记录，不再留下由序列化失败产生的半份COMPLETE JSON。
3. 每臂成功完成后单独保存`.ordinary.json`/`.aligned.json`指标备份，记录来源、缓存身份、200次完成更新及成本；它们仍是指标报告，不是权重checkpoint。artifacts包含已有主报告、progress与两臂备份的SHA。
4. 旧缓存兼容只接受原source manifest SHA `b92c8ea792b3066c5543c55163d47c7b1fa1a9634dbab8e393093fe9c61b4063`和旧runner SHA `38e3d88f73183b8e3d4f7f4831c606a19794e959c232d52bec9cc92ec4b943d3`的精确版本。旧manifest原文嵌入新manifest，并核验其SHA；52项绑定中仅runner允许这次变更，其余51项必须相同。B、protocol、TRAIN fixture/全标注身份仍按当前固定值精确比较。缓存manifest与payload身份须彼此相同，128视图顺序、各缓存文件SHA、检测器前后状态、GT/资格/CPU tensor形状等检查保留。
5. 主报告分别保存当前执行identity和原缓存identity；旧版复用标为`cache_action=reused_legacy_report_only_fix`。不改写旧缓存、截断主报告或历史SHA；当前版同身份缓存复用标为`reused_current_cache`。

原协议、64图fixture、G模型源码SHA仍分别为`777278e86a0598bc55b2d35cbfb5f79e0887574556888729b5c3c0d03862b9c2`、`616ac890b9a60689c7fb52c08bf58b4fe85e2542090b158d79ce0d429902209c`、`2545e32ef953d22ef359a56bfc1cadd38d28e69f22b30de6c7fa60f27092c51b`。新版runner SHA `45c5cd608f11364164efa0903d83606082e8066487453d472afd44f550fe0875`，新版manifest SHA `7840ab1c8d68d327d0ac6793d59b4e6cc0340d846087499f2d350692155dc200`；第48节v4及旧manifest身份保留为历史证据，本次服务器交付用本节修复版。

**本地复核已完成：** torch1.8.0.post3/Python3.8环境40项CPU回归检查通过，包含完整两臂2步合成入口＋support/逐帧/联合delta JSON读回、非零越界计数、NumPy类型保持、NaN/Inf/未知对象拒绝、禁止覆盖/原子发布失败清理、原/新身份128视图缓存复用、缓存不被改写、B/TRAIN/protocol/fixture/头/旧runner身份变化及缓存文件变更拒绝、最终序列化失败转FAILED。Python3.8语法、52项当前来源绑定、G文件diff空白检查通过。静态主报告`work_dirs/port_geometry_g_v1_report_fix_local_20261002/static_check.json`状态`STATIC_CHECK_COMPLETE_NO_GPU_NO_UPDATES`，检测器/头真实数据更新均0。回归缓存是CPU合成缓存；服务器实际旧cache文件完整性与复跑CUDA仍待验证，不能称已经恢复服务器结果。

### 50.4 服务器恢复指令（准备完毕，未执行）

上传这两个修改后的运行文件到原项目同名位置：

- `crane_project/tools/preflight_port_geometry_g_v1.py`
- `crane_project/tools/port_geometry_g_v1_sources.json`

测试文件可同步；无需重传未变的G模型/协议/fixture，也不制作压缩提交包。保留原失败输出、progress和已有`port_geometry_g_v1_roi_cache`，采用新输出名：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_v1.py \
  --check-only \
  --out-json work_dirs/port_geometry_g_v1_static_check_report_fix_v1.json
```

确认静态状态成功后复用原缓存：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_v1.py \
  --gpu 0 --reuse-cache \
  --out-json work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache
```

旧版完整缓存须显示`reused_legacy_report_only_fix`。这会跳过B特征提取，**仍重跑两臂共400次小头更新**；上次没有保存头权重或独立臂报告，不能仅“补写JSON”恢复全部数值。若原缓存不完整或SHA不符，应回传错误，不删保护或手改manifest。没有增加图像batch、头batch、ROI尺寸、训练预算或并行头，不引入新的模型图；具体显存仍看此次refiner成本字段，不保证绝对峰值不变。

成功应为`TRAIN_SHORT_FIT_COMPLETE_REVIEW_REQUIRED`、`formal_training=false`、`detector_updates=0`、`head_updates_total=400`、`checkpoint_exported=false`。回传新主JSON、同名progress/artifacts和ordinary/aligned备份；另回传旧cache_manifest及12张previews便于真实空间支持复核，不必传20MiB特征payload。依据第48.6节联合几何/冻结与坐标门槛判断下一阶段，不调整B、不读TEST调参。TEST已在前序工作多次暴露，本轮无新增VAL/TEST执行。

## 51. G v1完整短拟合回传：probe联合审查，不进入当前正式方案（2026-10-02）

### 51.1 回传身份与必要复算（事实）

用户回传静态主报告/progress/artifacts及TRAIN主报告/progress/artifacts/ordinary/aligned两臂备份，共8个文件。原字节已归档到`work_dirs/port_geometry_g_v1_server_review_20261002/`，一次性复算脚本为该目录`review_received.py`，机器可读分析为`review.json`；只分析既有TRAIN，不连接服务器、不训练、不读新VAL/TEST，未修改任何运行模型/损失/协议。

- 两份artifacts引用的全部回传文件SHA通过；静态progress为空是正常的静态模式。静态主SHA `b58d9be92cfe33bd7f60be61789ecec6216125028187703cb8d206700fc8c765`，TRAIN主SHA `ce38c0f399027de2a72bad8ff94fa5691cb1aa0d1e66048c8ef32d025487bc71`。静态与TRAIN当前identity相同，52项代码来源、protocol/fixture均匹配本地固定版。
- 状态分别为`STATIC_CHECK_COMPLETE_NO_GPU_NO_UPDATES`与`TRAIN_SHORT_FIT_COMPLETE_REVIEW_REQUIRED`；`cache_action=reused_legacy_report_only_fix`。原缓存身份通过第50节精确兼容检查；主报告中缓存manifest内容按原writer格式重编码所得SHA `7dae321ea2c0fb3f60eb4eb20cee9ebfb8f404d5c0780308485a24ed16f5108d`与两份artifacts声明相同，确认静态/拟合指向同一缓存。两臂备份的result、cost、当前/旧身份均与主报告相同，均完成200步，总新增头400步、B更新0、未导出拟合头权重。
- 初始模型SHA及两臂初始逐帧结果完全相同；128视图顺序及参考GT与固定fixture匹配，历史B重放容差通过。200个预生成batch与seed1703重建结果相同，每批4real+4sim、仅fit有效视图；probe未优化。20条progress快照与两臂主报告日志完全一致，不冒称400步完整裁剪日志。
- 按已保存GT/框复算中心、边长、角度、分组汇总及三种几何delta，确认中心/score/输出数保持、界限有效；回传参考GT与服务器解析GT的五分量差均为0。纯几何字段一致；本地与服务器RIoU相关复算最大差`4.329842947958795e-6`（主要是描述性GT替换gain），不是完全逐位相同。保留服务器原值作主证据，不修改成绩或门槛；独立NumPy原图SmoothL1复算与回传loss最大差`4.743797923922877e-8`。
- 三个首步任务梯度有效，step2 stem梯度有效，各臂报告finite检查通过。20条快照中clip multiplier均为1，最大记录norm ordinary .567772、aligned .752626，低于10；首步stem为0符合零最后层设计。最终平方投影0、三分量≥95%上限饱和计数均0，不支持把这次退化归为数值崩溃、已记录的大幅裁剪或残差饱和。

原cache_manifest单文件、CPU feature payload及12张实际previews尚未回传；其文件SHA/张量校验由服务器运行代码执行，本地没有独立重读这些实际文件。主报告包含完整缓存声明及来源，不能把此声明等同于本地实看空间采样图。

### 51.2 probe尺寸、纯角度与RIoU（事实，B/ordinary/aligned同框配对）

probe为real8图、sim8图，两个视图，共**16张独立图名、32个相关视图**；每行8视图。B已在TRAIN中使用全部图，角色只隔离G头的拟合，不是独立VAL或序列泛化。角度采用全部输出框的纯pi周期角误差，不以protocol惩罚或预测近方形筛选替代。

| probe分组 | 方法 | 长边平均绝对相对误差% | 短边平均绝对相对误差% | 纯角度RMSE° | 纯角度p90° | 全帧平均RIoU |
|---|---|---:|---:|---:|---:|---:|
| real原尺度 | B | 3.6494 | 3.6011 | 1.4333 | 2.3732 | .899582 |
| real原尺度 | ordinary | 3.8527 | 3.9133 | 1.3087 | 2.0459 | .897322 |
| real原尺度 | aligned | 3.9095 | 4.0332 | 1.3610 | 2.1368 | .896429 |
| real半尺度 | B | 4.6231 | 5.2789 | 4.8263 | 7.4295 | .851549 |
| real半尺度 | ordinary | 4.7433 | 4.4352 | 5.0104 | 8.1347 | .850575 |
| real半尺度 | aligned | 4.8359 | 5.1678 | 4.8866 | 7.5126 | .851231 |
| sim原尺度 | B | 1.8690 | 1.0052 | 1.1984 | 1.8887 | .939004 |
| sim原尺度 | ordinary | 1.8921 | 1.3521 | 1.2266 | 2.1272 | .935460 |
| sim原尺度 | aligned | 1.9171 | 1.4050 | 1.2475 | 1.8745 | .934464 |
| sim半尺度 | B | 7.8345 | 5.8742 | 1.5627 | 2.1961 | .812432 |
| sim半尺度 | ordinary | 5.3767 | 5.2381 | 1.8919 | 2.5310 | .811843 |
| sim半尺度 | aligned | 6.0806 | 4.9060 | 1.7650 | 2.2856 | .811386 |

观察：real原尺度是角度略好但两边尺寸与RIoU退化；real半尺度短边略好、长边及角度RMSE/p90退化，RIoU略退；sim原尺度两臂边长/角度RMSE/RIoU均退；sim半尺度尺寸有改善，但角度RMSE/p90及RIoU均退。两臂四组RIoU全部低于B；aligned相对ordinary仅real半尺度RIoU较好，其余3组较差，没有稳定旋转对齐增量。不能事后按域/尺度拼接选头或换step。

两臂probe三口径均为输出覆盖32/32=100%、仅输出中心正确32/32=100%、全帧中心正确32/32=100%；全部固定128视图也均128/128。这只是当前稀疏样本且中心复制B的事实，不能外推成real全视频100%，也不能验证漏检帧、已有错位输出或真实连续性保护。普通B的real VAL360/375全帧中心正确覆盖仍是既有事实。

**fit与probe对比：** 两臂4个fit组原图loss均下降；96 fit视图总体B长/短边误差3.5252%/3.4849%、角度RMSE2.9595°、RIoU.871079，ordinary为2.9009%/2.4560%、2.6323°、.877141，aligned为3.0034%/2.8975%、2.6005°、.875236。但32 probe总体B为4.4940%/3.9399%、2.7030°、.875642，ordinary为3.9662%/3.7347%、2.8240°、.873800，aligned为4.1858%/3.8780%、2.7569°、.873377。总体尺寸均值改善主要来自半尺度sim，不能覆盖逐组退化。原尺度real/sim probe的原图loss两臂均上升；半尺度probe loss下降亦未转化为RIoU收益。

配对计数：ordinary的32个probe视图RIoU改善12、退化20；aligned改善10、退化22。三个绝对误差同时降低且RIoU升高的严格逐视图联合改善计数为ordinary1/32、aligned0/32；这是补充描述，并非新增或更改第48节的正式放行门槛。每组只有8视图，部分尾部由少数样本影响，不能称统计显著退化、稳定收益或已证明全量训练必失败。

### 51.3 保存框机制补核：分量更准不必然有更好重叠（事实与推断分开）

复用已保存框，额外计算“B中心/角度＋G尺寸”与“B中心/尺寸＋G角度”两个描述性混合框，未拟合新模型、未改变任何方法参数。这些RIoU变化是非加性的保存框替换，不是可部署消融或学习因果归因。

| probe分组 | ordinary仅尺寸ΔRIoU | ordinary仅角度ΔRIoU | aligned仅尺寸ΔRIoU | aligned仅角度ΔRIoU |
|---|---:|---:|---:|---:|
| real原尺度 | -.002667 | +.000081 | -.003476 | +.000175 |
| real半尺度 | +.000885 | -.001613 | -.000568 | +.000458 |
| sim原尺度 | -.002940 | -.000549 | -.003751 | -.000687 |
| sim半尺度 | -.002144 | +.000477 | -.000854 | -.000384 |

尺寸修正本身已可降低RIoU，不能将退化全部归为角度。具体例子是半尺度`sim_seq08_00168`：B中心误差7.1989px保持不变，原GT短边19.2844px；ordinary长边误差16.8645%→4.3956%、短边13.7110%→3.3280%，纯角度.9777°→2.4444°，RIoU.527094→.511584。只套用ordinary尺寸、保留B角度，RIoU仍为.514731，低于B；这例支持“已有中心错位下尺寸更接近GT也可能减少交叠”的具体几何关系，不能推广成所有框尺寸变准都会变差，也不据此扩大预测框。

**事实：** 原G loss只优化归一化长边/短边/角度误差，没有直接RIoU或中心纠偏；固定中心不能处理定位距离。support mask记录全部均值1、半尺度GT短边平均约2.1～2.2个P3单元，边长目标无超界，少数real角度目标超过10°但最终无上限饱和。无证据需要通过扩大残差上限、放松覆盖或增加loss权重来“救”这次结果。

**推断/未确定原因：** fit获益而probe未形成联合获益，与小样本纠偏迁移不足或头容量/表征限制相容；分量代理目标与中心固定的重叠耦合确实存在。尚不能确定哪项主导，也不能证明P3缺乏边界信息、旋转采样必然无效、过拟合是唯一原因或D/E-H/F-S的原始根因已经查明。

### 51.4 阶段判断、下一项有限检查与显存（建议及待验证）

**结论：不推荐按当前G v1直接进入正式全TRAIN对照，保留B ep24。** 工程/有限更新检查通过，联合几何目标未通过；这是按原预定条件判断，不据TEST调参，也不是宣告局部细化路线永久关闭。不能用末批loss下降、整体尺寸均值下降或“全量数据可能更好”替代当前四组证据。

先补回已存在的cache_manifest及12张空间采样预览，无须新的B推理或GPU拟合。下一项若继续，建议固定一项**ROI信息对应消融**，检验是否学到与当前图像对应的局部几何纠偏：复用现有缓存，比原ordinary ROI与同role/domain/scale内固定置乱ROI（mask随ROI移，保留本帧B描述符与GT目标）；两臂头/初始化、loss、200步、batch/seed及界限不变，图级两个尺度保持同一固定供体对应，fit/probe供体不混用。此方案还需授权实现和拟合前合同固定，本轮未执行。它检验局部信息对应的作用，不是证明ROI绝对无用或修复全部几何问题。即使真实ROI胜过置乱，也仍须满足原probe联合目标后才能支持正式阶段；不搜索step、混合两臂或同时改容量/权重/中心项。

服务器成本记录：原采集B前向最大allocated343.3833MiB、reserved412MiB；原局部采样最大allocated172.9233MiB、reserved412MiB；本次缓存复用后两个小头阶段peak allocated均5.3311MiB、reserved6MiB，普通/对齐总阶段耗时约6.34s/5.68s。这里是PyTorch阶段统计，不是nvidia-smi进程总占用，不含CUDA上下文/其他库，也不能保证未来全TRAIN/多GPU训练的峰值。原采集runtime记录物理可见卡0，本次拟合可见卡3；缓存特征身份/原数值重放均已核对，不要求相同物理卡。

### 51.5 后续回传打包命令（用户本轮授权，可只传一个压缩包）

为减少文件上传操作，打包本次8个报告及仍缺的cache_manifest/12张previews；不包含20MiB`local_roi.pt`、B权重或其他实验目录。本轮只给指令，没有在本机或服务器自动生成包。采用时间戳新包名并保留原件：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
bundle="work_dirs/port_geometry_g_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$bundle" -C work_dirs \
  port_geometry_g_v1_static_check_report_fix_v1.json \
  port_geometry_g_v1_static_check_report_fix_v1.progress.jsonl \
  port_geometry_g_v1_static_check_report_fix_v1.artifacts.json \
  port_geometry_g_v1_train_preflight_report_fix_v1.json \
  port_geometry_g_v1_train_preflight_report_fix_v1.progress.jsonl \
  port_geometry_g_v1_train_preflight_report_fix_v1.artifacts.json \
  port_geometry_g_v1_train_preflight_report_fix_v1.ordinary.json \
  port_geometry_g_v1_train_preflight_report_fix_v1.aligned.json \
  port_geometry_g_v1_roi_cache/cache_manifest.json \
  port_geometry_g_v1_roi_cache/previews
printf '%s\n' "$bundle"
```

后续上传该包即可集中检查，不再要求逐个上传同一套报告。B仍固定，TEST已多次暴露，本轮未使用其任何新结果或进行选权。

## 52. G ROI信息对应消融：固定实现、代码复核与服务器指令（2026-10-02）

### 52.1 本轮授权与问题（事实/待验证分开）

用户授权按第51节下一步建议实现并给服务器/压缩指令。新增独立有限TRAIN入口，不改原G文件、B/D/E-H/F-S配置或可靠性分支。未连接服务器、未启动真实缓存拟合或正式训练，也未生成本地提交文件压缩包。

**待验证问题：** 在保留本帧B框描述符和原图目标的条件下，正确对应本帧的局部ROI，是否比同分布但错配的ROI提供可复用的纠偏信息？这个检查帮助区分信息对应作用与通用框描述符/共享校正，不能直接证明P3信息缺失、过拟合是唯一根因或某个新方法已有效。matched严格重放第51节已有ordinary结果，因此本消融不可能把该已失败的当前G v1重新包装为通过联合门槛；机制证据只用于讨论后续设计。

### 52.2 拟合前固定合同（固定设计）

- 条件为`matched`（本帧ordinary ROI＋mask）和`shuffled`（供体ordinary ROI＋随附mask）。两臂仍使用原183,907参数G头、相同初始化、原图长/短边/pi周期角loss、Adam及所有界限；没有aligned臂、中心补偿、分类/质量头、教师、DINO或候选排序。
- 原seed1703、Adam lr.001、weight_decay0、clip10、batch8（4real＋4sim）、每臂200次更新保持，合计新增头400步；相同200个batch须与第51节基线精确一致。仅报告初始与step200，不选step或搜索seed/超参，不导出头checkpoint。B前向/更新均0，只读既有CPU特征。
- 置乱seed独立固定为1704。按固定fixture顺序分别对fit-real24、fit-sim24、probe-real8、probe-sim8进行无自供体的双射置乱；拒绝自身对应只为生成预声明derangement，不读取GT或结果选择映射。完整64图供体map已在protocol中写死并与确定性生成结果核对；同一图的clean/half使用同一供体图，取供体的相同尺度。不同角色/域/尺度不混用，不把probe特征用于fit。
- **仅移动ROI与mask。** 本帧模型坐标B、原图B及score、GT参考、解析GT、frame/role等元信息原样保留；框描述符仍由本帧B模型框计算，供体框/score/GT从不替代本帧输入或目标。domain/role只用于离线配对/批次，不作为头输入；GT只进入独立loss。
- 数据仍为64张TRAIN、128视图，fit48图/96视图、probe16图/32视图。此精确已审查缓存128视图均有真实B输出且中心资格有效；一旦替换缓存导致缺输出/不合资格，停止，不能临时筛选样本、GT造框或改变分母。
- 两臂复用原G的`fit_arm`/`evaluate`，通过独立record视图新增matched/shuffled局部键；CPU张量只是共享引用，没有复制整套ROI或整张FPN。原dataset等比例变换、raw w/h还原、原图loss与深度约束均未改变；复制中心不能改善定位距离，改变尺寸/方向仍不能保证深度精度。

**对照局限：** 单次同域同尺度置乱可能仍保留同视频/同类物体信息，且同时改变ROI图像内容、上下文及其与本帧几何描述的对应。它不是纯外观因果隔离或真实部署模块。真实ROI优于置乱只是相对机制线索；相对无增益也不能证明所有图像信息无效。probe仍来自TRAIN、B见过全部图，不是独立验证或序列泛化；当前TEST已多次暴露，本轮无VAL/TEST访问或选权。

### 52.3 实现、重放保护与来源（事实）

新增四个文件：

1. `crane_project/tools/preflight_port_geometry_g_roi_ablation_v1.py`：独立静态/有限拟合入口、固定置乱adapter、原ordinary重放、两臂原图评价及归档。
2. `crane_project/tools/port_geometry_g_roi_ablation_v1_protocol.json`：完整供体映射、原G超参、介入/数据/评价边界与预算。
3. `crane_project/tools/port_geometry_g_roi_ablation_v1_sources.json`：原G52项SHA全部保留，追加原G source manifest及新入口/协议，共55项绑定；原合同不改，新manifest自身不循环绑定。测试不是服务器运行依赖。
4. `tests/test_port_geometry_g_roi_ablation_v1.py`：必要CPU配对、数据保留、重放及完整入口回归。

入口要求第51节完整TRAIN主报告SHA `ce38c0f399027de2a72bad8ff94fa5691cb1aa0d1e66048c8ef32d025487bc71`、原G source manifest SHA `7840ab1c8d68d327d0ac6793d59b4e6cc0340d846087499f2d350692155dc200`、真实cache_manifest SHA `7dae321ea2c0fb3f60eb4eb20cee9ebfb8f404d5c0780308485a24ed16f5108d`。保留原缓存128视图顺序、文件SHA、CPU tensor/GT/资格/形状和冻结状态核验；实际缓存B原图框/score/GT等需与旧ordinary初始逐行一致。没有旧特征时停止，不重新跑B采集或拼接新缓存。

CUDA拟合前核对基线Python/torch/CUDA/cuDNN/NumPy/OpenCV版本字符串（允许物理卡可见编号不同）。matched先跑200步并保存备份，规范日志条件名后，完整result必须与基线ordinary逐项相同，包括模型SHA、初/末框、loss/梯度快照及汇总；不符则FAILED并停止在shuffled开始前。已有结果目录拒绝覆盖；主报告和两臂备份用原G已验证原子JSON writer，序列化失败留有效FAILED及已完成证据。

正常结果目录包括`completion.json`、`progress.jsonl`、`artifacts.json`、`matched.json`、`shuffled.json`、`donor_assignment.json`、protocol/source副本、cache_manifest副本及12张`previews/`PNG。原缓存/旧报告保持，特征payload/B权重不复制到结果目录。12张图保持原G的GT绿色/B黄色/ordinary青色/旧aligned品红图例，**品红点不是本次shuffled**；本次只用ordinary，错配供体由donor_assignment追溯。artifacts绑定结果文件的相对路径和SHA，便于整个目录打包回传。

当前新入口SHA `1c8f2c04ac3000c5b7ce0e06a71202fb5944d121350efbe57b6b5227cef8afba`，协议SHA `84f39b8050fd1b57b125516f0d7aa763fca9bcd6475c6be9ffc786905adfc88e`，新source manifest SHA `0250eafb9feef5fff88a4072dca02b1a0cb71f345f6fa1e0b866c2d8a2ada170`。

### 52.4 本地检查与实际边界（事实）

Python3.8/torch1.8.0.post3环境**15项CPU回归通过**：64图固定映射/双射/无自身与角色域隔离；两尺度同供体；只移动ROI/mask且无整套张量复制，框/GT/描述符保留；错配/缺输出/资格/重复视图拒绝；合成两步拟合真实ROI精确重放、置乱梯度有效与中心/score保护；完整报告JSON读回；source/runtime变化拒绝；静态入口输出/禁止覆盖；序列化失败有效FAILED；完整入口成功归档12个模拟预览以及重放失败时仅完成matched、没有shuffled文件。完整入口测试明确mock了CUDA调度并实际跑CPU合成两步，不是服务器CUDA或真实200步实验。

实际本地静态入口通过，使用归档原TRAIN报告原字节、当前64图/全TRAIN标注身份和供体合同；报告`work_dirs/port_geometry_g_roi_ablation_v1_static_local_20261002/completion.json`，状态`STATIC_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`，真实B/头更新均0。该静态模式不读取feature payload或使用GPU。本机没有该实际ROI缓存及CUDA环境，真实cache读取、200步精确重放/置乱收益和新显存峰值仍待服务器验证。

代码复核确认没有检测器实例化/特征采集分支，不向头输入供体GT/box或改变本帧descriptor；原52项G依赖SHA保持。Python3.8语法、55项来源绑定及diff空白检查通过，下面命令以`bash -n`核验语法，未执行。命令副本位于`work_dirs/port_geometry_g_roi_ablation_v1_local_review_20261002/server_commands.sh`。不重复原40项同版本检查或完整几何审计。

内存设计仍是既有20.329MiB CPU ROI tensor＋同批8的小头，每个条件串行拟合后释放；没有新增B/FPN图、图像batch、ROI分辨率或同时驻留两头。新record字典/JSON参考仍有CPU元信息开销；实际显存不能以设计或CPU测试保证，服务器会记录refiner阶段峰值。

### 52.5 服务器运行、判读和压缩（已准备，未执行）

上传上述前3个运行文件至服务器同名位置，第4个测试可选；原G、基线主报告和ROI缓存保持原位置。先激活原环境并做静态合同检查，输出目录须不存在：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_roi_ablation_v1.py \
  --check-only \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --out-dir work_dirs/port_geometry_g_roi_ablation_v1_static
```

确认`STATIC_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`后进行两臂有限检查（物理卡3、逻辑0）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_roi_ablation_v1.py \
  --gpu 0 \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache \
  --out-dir work_dirs/port_geometry_g_roi_ablation_v1_train
```

成功状态`TRAIN_ROI_ABLATION_COMPLETE_REVIEW_REQUIRED`，`matched_reference_replay=EXACT_REPLAY_PASS`、`head_updates_total=400`、B前向/更新0、无checkpoint导出。若FAILED，保留并回传该结果目录，不手改SHA/版本保护、重抽映射或继续增加步数。已存在目录换新名，压缩命令相应更换，不删除旧证据。

判读先检查重放/机械条件，再逐fit/probe/domain/scale比较尺寸、纯角RMSE/p90及RIoU；`matched_minus_shuffled_final`为matched减shuffled，RIoU正向、误差负向为相对改善。三口径输出覆盖、仅输出中心正确、全帧中心正确及全部稀疏视图保留。不以仅loss、整体均值或胜过置乱替代B联合目标，不统计伪连续性指标、不自动启动当前G正式训练。

压缩静态和拟合目录，内含预览和所有结果，可以只回传一个包：

```bash
bundle="work_dirs/port_geometry_g_roi_ablation_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$bundle" -C work_dirs \
  port_geometry_g_roi_ablation_v1_static \
  port_geometry_g_roi_ablation_v1_train
printf '%s\n' "$bundle"
```

该包不含原B权重、旧`local_roi.pt`或其他实验目录。本轮只提供服务器压缩命令，未在本地生成包。TEST已多次暴露，原VAL选权、B ep24与第51节当前G不正式训练的结论均保持。

## 53. G ROI对应消融回传：局部信息有相对作用线索，当前G仍未通过联合目标（2026-10-02）

### 53.1 收到文件、来源和必要代码核对（事实）

用户回传`/Users/mac/Downloads/port_geometry_g_roi_ablation_v1_review_20261002_202014.tar.gz`及静态/拟合终端。包SHA `75fcdb9a2d042cf750f3e9a632e334a02d6f6934cfcb09e7eeb0da343ef55897`，1,782,603字节，26个普通文件，无链接；校验路径后保持原字节归档在`work_dirs/port_geometry_g_roi_ablation_v1_server_review_20261002/`。只分析已生成TRAIN结果，不连接服务器、不新增B推理/头训练、不读VAL/TEST。本轮未修改检测、细化、损失、协议或可靠性源码；分析脚本与记录属于本轮交付。

机器可读复核为该目录`review.json`，SHA `33961863996b5d086034c8f01e87ec0bf548b4c6635a8fea14b1df9924cea3a1`；一次性`review_received.py`调用第51节既有保存框数学核对函数，不启动模型更新。完成以下必要检查：

- 两份artifacts所绑定的24个文件SHA及完整文件清单通过，另两个文件为artifacts自身。TRAIN completion SHA `e5ad4bf9ffcfc54993469b1c5810bd6e27bc87aa70574f0bb21670789c4275bd`；静态completion SHA `8a9b72f8705bbe41697969efb17c9dee1b730f43fc215375622708664ffcf848`。静态progress空文件正常。
- 当前55项运行来源、固定TRAIN标注/64图fixture、协议与第51节baseline身份匹配本地版本。真实cache_manifest SHA `7dae321ea2c0fb3f60eb4eb20cee9ebfb8f404d5c0780308485a24ed16f5108d`与原报告相同；12张PNG的实际字节SHA逐个匹配原缓存声明。**原feature payload没有回传**，因此本地没有独立重读ROI数值/方差或最终头权重；服务器执行的缓存校验和本地实读manifest/PNG须区分。
- 静态状态`STATIC_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`；拟合状态`TRAIN_ROI_ABLATION_COMPLETE_REVIEW_REQUIRED`，两臂均200步、合计400步，B前向/更新0、无拟合checkpoint导出。终端未粘贴最后Saved行，但归档completion/artifacts均完整，不能据终端截断认为未保存。
- 本地再次将matched完整result规范日志arm名后与第51节ordinary逐项比较，**精确重放通过**，包括头初/末参数SHA、128视图初/末框、loss/梯度快照和全部汇总，而非只有终端loss相同。两臂初始参数与初始B结果相同；备份与主报告result/cost/identity一致；20条progress快照与对应日志相同。200个4real+4sim的batch与旧baseline及seed1703重建结果相同，没有probe优化。
- 64图固定无自身双射、128个供体视图均匹配seed1704预声明映射；fit/probe、域、尺度隔离且两个尺度使用同供体图。静态代码核对确认adapter只改ROI/mask引用，本帧box/descriptor/score/GT保持；没有供体GT在线输入。单次置乱仍保留同域/同尺度、sim同视频等相似内容，不能解释成完全无图像信息对照。
- 按保存框独立复算中心、边长、周期角、分组汇总和delta，确认方向、分母、中心/score/输出数、界限均正确；GT参考与服务器解析五分量差为0。与服务器RIoU相关复算最大差`4.329842947958795e-6`，独立NumPy原图SmoothL1复算最大差`5.843678528572038e-8`；保留服务器原值，不放松成绩门槛。RIoU容差仅用于跨环境浮点复核。

首步三个任务梯度有效、step2 stem梯度有效；finite检查通过。20条梯度快照clip multiplier均1，记录norm最大matched .567772、shuffled .546354，均低于10；最终投影0、三个残差饱和计数均0。没有发现足以解释失败的重放/供体/分母/梯度数值错误，不把有限快照称作完整400步裁剪轨迹。

### 53.2 probe逐域逐尺度：相对置乱和相对B是不同问题（事实）

probe16个TRAIN图名、32个相关视图，每行8视图，B已见过全部图。边长为平均绝对相对误差，角度为全部输出的纯pi周期误差，不混用近方形筛选或协议中心惩罚。matched就是此前ordinary，不是新的更优权重。

| probe分组 | 方法 | 长边误差% | 短边误差% | 纯角RMSE° | 纯角p90° | 全帧mean RIoU |
|---|---|---:|---:|---:|---:|---:|
| real原尺度 | B | 3.6494 | 3.6011 | 1.4333 | 2.3732 | .899582 |
| real原尺度 | matched | 3.8527 | 3.9133 | 1.3087 | 2.0459 | .897322 |
| real原尺度 | shuffled | 4.2609 | 4.5883 | 1.3750 | 2.1972 | .890474 |
| real半尺度 | B | 4.6231 | 5.2789 | 4.8263 | 7.4295 | .851549 |
| real半尺度 | matched | 4.7433 | 4.4352 | 5.0104 | 8.1347 | .850575 |
| real半尺度 | shuffled | 5.8300 | 4.8310 | 4.6814 | 6.9936 | .848906 |
| sim原尺度 | B | 1.8690 | 1.0052 | 1.1984 | 1.8887 | .939004 |
| sim原尺度 | matched | 1.8921 | 1.3521 | 1.2266 | 2.1272 | .935460 |
| sim原尺度 | shuffled | 2.6150 | 2.9382 | 1.4460 | 2.2684 | .921695 |
| sim半尺度 | B | 7.8345 | 5.8742 | 1.5627 | 2.1961 | .812432 |
| sim半尺度 | matched | 5.3767 | 5.2381 | 1.8919 | 2.5310 | .811843 |
| sim半尺度 | shuffled | 7.0516 | 5.7751 | 1.4310 | 2.1423 | .823813 |

**相对置乱：** matched四组长/短边平均误差均更低；RIoU在real原/半尺度与sim原尺度更高，差分别+.006848、+.001670、+.013765，但sim半尺度为-.011969。角RMSE/p90在两个原尺度组更好，在两个半尺度组更差。总体32视图matched相对shuffled的RIoU改善20、退化12，三个分量绝对误差同时降低且RIoU升高10/32；相对优势不是逐组联合稳定优势。

**相对B：** matched四组RIoU全部降低，仍为12视图改善、20视图退化；严格逐视图联合改善1/32。shuffled只有sim半尺度RIoU高于B，其他3组更低，整体9视图改善、23视图退化、严格逐视图联合改善3/32；该臂是机制对照，不是可选推理方案，不能按域/尺度拼接两臂或借本报告改选step。严格计数只是补充描述，没有新增/改变第48节放行门槛。

probe总体B/matched/shuffled长边误差4.4940%/3.9662%/4.9394%，短边3.9399%/3.7347%/4.5332%，角RMSE2.7030°/2.8240°/2.6431°，RIoU .875642/.873800/.871222。总体角RMSE与p90方向也可不同，不能用一个整体数掩盖逐组结果。

三口径分别为输出覆盖32/32=100%、仅输出中心正确（<15px）32/32=100%、全帧中心正确32/32=100%；128视图也分别128/128=100%。这是复制B中心/输出及稀疏样本资格的事实，不代表全视频real正确覆盖100%或连续性已改善。普通B real VAL仍为输出374/375、输出条件中心正确360/374、全帧中心正确360/375；固定中心不能改善原中心距离、恢复无输出或修复已有严重错位。本轮不报告稀疏样本的伪视频时序指标。

### 53.3 末批loss、fit迁移与尾部（事实/推断分开）

终端step200 loss matched .0773883、shuffled .0737507是同一末批的优化前loss，不是完整fit/probe均值，不用于选方案。step1 loss相同符合零最后层的初始精确B；之后不同ROI带来的参数梯度不同，不是置乱未生效。

| 完整角色均值 | B | matched | shuffled |
|---|---:|---:|---:|
| fit原图归一化loss（96视图） | .1220326 | .0933661 | .0941102 |
| probe原图归一化loss（32视图） | .1390209 | .1300457 | .1524336 |
| fit RIoU | .871079 | .877141 | .879257 |
| probe RIoU | .875642 | .873800 | .871222 |

两臂四个fit组loss均低于B，置乱在3个fit组RIoU还高于matched；然而其probe原尺度real/sim更差。**推断：** 这与小样本头学到共享纠偏、框描述符规律或固定ROI/供体关联、迁移不足相容，不能证明哪种机制主导，也不能只凭fit成绩证明真实局部几何被学到。matched的尺寸相对优势支持保留局部对应信息的研究价值，不支持断言P3完全无信息。

半尺度sim的反例不是只由1个末批造成：matched相对shuffled RIoU为2视图改善、6视图退化。最大负差为`sim_seq08_00432`，.879679→.798812（-.080867），角误差.2221°→1.9761°、长/短边误差7.7710%/4.1078%→11.6873%/9.0773%；这是相对置乱的描述，不据此挑选其他模型。旧例`sim_seq08_00168`也保留：B中心7.1989px不变，matched把尺寸16.8645%/13.7110%降为4.3956%/3.3280%，但角误差.9777°→2.4444°、RIoU .527094→.511584；shuffled尺寸误差仍14.3948%/9.8745%却RIoU .524742。第51节“固定中心时尺寸分量更准不保证交叠更好”的几何关系仍存在，不能据这一消融将全部失败归为角度或容量。

### 53.4 真实采样图与资源缺口闭合（事实及局限）

已实际读取12张原缓存PNG，并生成两张分析联系图：本归档目录`real01_05_06_support_contact_sheet.png`及`real12_13_sim_support_contact_sheet.png`。它们只是显示已有图片的拼接，没有新的特征提取或头预测。每个TRAIN序列首个fit图的两尺度各一张；不是probe全体，也不是新两臂预测比较。

图例GT绿色/B黄色/ordinary青色/旧aligned品红：品红不是本次shuffled。预览可见网格落在框所在结构及邻近背景；sim倾斜框上ordinary轴对齐格与旋转GT方向不一致，而旧aligned随框旋转。不能从投影点图证明每帧所有标注端点均被采到、上梁边界特征足够，或aligned就应有效；旧aligned的probe联合结果第51节已失败。

清晰/半尺度图都画回原图，因此空间网格看起来相似不表示有效P3信息分辨率相同。缓存声明各support mean均1，只说明这些取样点未触及padding；不是结构命中率。原统计半尺度GT短边约2.1～2.2个P3单元，9×9双线性格不会自行产生额外独立特征细节。payload未上传，不能检查特征退化是否主因。第51节未实看manifest/PNG的缺口已闭合，但不能追改历史记录成当时已收到。

本次matched/shuffled小头阶段peak allocated均5.3311MiB、peak reserved均6MiB，耗时约5.82s/5.03s；既有CPU ROI21,316,608字节。该运行没有新增B图，显存与上一有限检查相同；统计不含CUDA上下文/库的进程总显存，不能保证未来正式训练资源峰值。MMCV打印是既有版本提示，本次结果没有显示其导致中断。

### 53.5 下一步判断与一项有限改进建议（建议，尚未实现）

**保留B ep24，不进入当前G v1正式全TRAIN。** 本消融已回答对应关系相对作用这一有限问题，没有把matched从已失败的ordinary变成新赢家；不继续搜供体seed/训练step、扩大界限、按域拼头，或仅凭末批loss启用置乱。

若继续局部几何路线，优先提出一项**小容量、保留空间布局的残差头对照**，检验当前fit收益能否更好地迁移到probe。候选只将`hidden_fc=64`收为固定16：保留9×9 ROI及展平空间顺序、32通道stem、同帧4维B描述符、三维零初始有界残差；参数按现结构由183,907降为59,107，移除主要大FC容量，不增加新的图像分辨率/特征来源。16只是一个建议的预声明对照点，不是已验证最优容量，不在结果后搜索更多宽度。

其余输入缓存、TRAIN角色/200个batch、seed、Adam/loss/200步预算、原图还原/尺度/深度约束和联合目标均保留，原matched完整重放结果作为已审查对照。不同形状头不能声称所有参数初始化相同；应固定共同stem的初始参数、各自FC确定性初始化、零最后层及初始精确B。无需重复B采集、aligned和置乱两臂；实施前另锁定这个唯一改动的合同，再按授权做本地代码与有限检查。

**针对性及局限：** 它只检验“当前大展平FC的拟合能力与有限TRAIN迁移是否匹配”，不能宣称已证明过拟合根因，也不会解决中心固定与RIoU非线性耦合，不能恢复漏检或保证方向、时序及深度精度。若联合probe条件仍不满足，应结束当前小容量尝试并据实际缺口重新设计，不靠加步数/改权重反复试出过线结果。受限中心细化或新增结构监督均是另一个设计，不能混入这项单变量对照。

本轮分析/下一步建议并非新训练授权或正式放行，尚未改运行代码。可靠性路线继续由另一对话负责。TEST已多次暴露，本轮没有使用任何新TEST或VAL结果决定参数、方法、阈值或权重；原17项后续VAL条件及B的历史选权身份保留。

## 54. 单一FC容量64→16对照：固定实现、必要复核与服务器指令（2026-10-02）

### 54.1 最新授权、记录偏好及实验身份（事实/待验证）

用户授权按第53节建议实施小容量头对照，并说明`work_dirs/.../review_received.py`这类文件无需在本地继续保存。已清理用户指明的`work_dirs/port_geometry_g_roi_ablation_v1_server_review_20261002/review_received.py`；第53节关于其存在/执行的记载保留为历史事实，现不能再将它当可点击的当前脚本。未批量删除原回传输入或其他工作文件。本轮临时静态/CPU测试报告使用系统临时目录并在退出后清理，没有新增work_dirs复算脚本、报告或命令副本；长期执行记录集中在本文。

**问题与边界：** 检验当前大展平FC容量是否限制有限TRAIN纠偏的迁移；不是已经证明过拟合根因，不解决中心固定与交叠耦合，也不证明后续VAL收益。本次新增独立入口和小头，未修改原G/ROI消融、B/D/E-H/F-S或另一对话的可靠性源码，未连接服务器、未启动真实GPU检查或正式训练。B仍为VAL固定ep24，TEST已多次暴露。

### 54.2 已固定的唯一介入与初始化（固定设计）

- `hidden_fc`由64变为16，参数183,907→59,107；仍为32通道原stem、9×9空间顺序展平＋同帧4维B描述符、三维有界logL/logS/pi周期角残差。小头直接继承原`LocalGeometryRefiner.forward`，没有池化、分辨率、损失权重、采样位置、特征来源、中心、分类/score、分配或输出过滤改动。
- 原ordinary匹配ROI/mask与128视图CPU缓存复用。图级fit48图/96视图、probe16图/32视图；B已见过这些TRAIN图。仅小头新增**200次**更新，原seed1703的同200个4real+4sim批次、Adam lr.001/wd0/clip10、原SmoothL1三分量归一化及界限保持。不重跑原64、aligned或shuffled，不新增B前向，不搜索宽度/step/seed，不导出拟合checkpoint。
- 64宽度参考直接读取第51节原完整ordinary报告（SHA `ce38c0f399027de2a72bad8ff94fa5691cb1aa0d1e66048c8ef32d025487bc71`），第53节matched已精确重放它。新报告完整附带这个历史参考，明确`additional_reference_updates=0`，不能把历史200步加到本次预算。
- 重新按seed1703在CPU构造**未训练**64头，完整初始参数/缓冲摘要必须等于参考初始摘要；随后16头拷贝相同stem、预先固定的前16个hidden单元权重及bias、对应零输出权重/bias。不是加载已训练头做剪枝，没有按GT/效果挑通道；未声称所有不同形状参数都相同。两头的初始输出均须精确B。
- 真正GPU检查要求原Python/torch/CUDA/cuDNN/NumPy/OpenCV版本字符串相同；物理卡编号可不同。原B源、图/标注、缓存manifest/payload/PNG文件SHA、张量维度、GT、128行顺序、资格与原初始B逐行保持。缺缓存或摘要不匹配时停止，不重新采集或跳过保护。

评价仍报告每个fit/probe/domain/scale的长/短边mean/p90、纯角RMSE/p90、RIoU与尾部、仅输出中心命中、输出覆盖、全帧中心正确覆盖。`fc16_minus_b`及`fc16_minus_fc64`都是新16头减对应参考，误差负、RIoU正表示改善；只报告初始与固定200步。稀疏TRAIN不报告伪连续性；尺寸/方向改变不保证深度精度。沿用第48节联合判断与未来原17项VAL条件，不因优于64头或fit loss降低自动进入正式阶段。

### 54.3 本地代码、来源和必要检查（事实）

新增4个服务器运行文件及1个本地测试：

1. `crane_project/utils/port_geometry_refine_g_capacity_v1.py`：唯一16宽度构造与共同初始参数拷贝，SHA `9b90d258020074cbd77b2227aafb141f54578e9e0fdb8197ef74d2fac260f908`。
2. `crane_project/tools/preflight_port_geometry_g_capacity_v1.py`：静态合同/缓存复用/单臂有限检查/原图评价及原子报告，SHA `903fedd7ace2ef0f54bc266b9d659a45fbbe78d94f79990ce77f64ff04b29cf9`。
3. `crane_project/tools/port_geometry_g_capacity_v1_protocol.json`：固定身份、初始化、预算及评价合同，SHA `7ca5d7dd24443b923eae1d92ec7bef550da59d0efe03ecf56dcfa9f91d7606be`。
4. `crane_project/tools/port_geometry_g_capacity_v1_sources.json`：原55项全部保持，追加原ROI消融source manifest自身和本次工具/小头/协议，共59项绑定；本manifest自身不循环绑定。SHA `39076647989f1a5d71b43434719a668c05f5e8d4f2e663b4d6e8885729ca3586`。
5. `tests/test_port_geometry_g_capacity_v1.py`：服务器运行不依赖此测试。

**14项CPU回归通过（0.50s）：** 唯一宽度/参数量与原forward继承、共同stem/前16单元完全对应且可重复、错误/学习后/非有限初始化拒绝；对原fit函数AST核对，仅替换头构造（optimizer/loss/evaluate/裁剪/梯度日志体一致）；实际合成CPU两步三任务梯度和step2 stem有效、中心/score与输入保持；在线输入/GT均不收梯度；JSON读回、来源变化拒绝、静态无cache/GPU/初始化、输出目录禁止覆盖；序列化失败有效FAILED；完整入口mock CUDA调度但实际执行合成两步CPU拟合的成功/初始对比失败分支，失败保留已完成的小头备份。该完整入口测试将预算显式mock为2，不冒称真实TRAIN200步或CUDA验证。

实际本地静态入口用原报告原字节和本机固定TRAIN运行，状态`STATIC_CAPACITY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`，59项来源/原fixture/标注/协议通过，B与头更新均0；临时结果已清理。Python3.8语法、59项SHA/原55项不变、diff空白及下列shell语法复核通过，未重复原G/ROI同版本全套检查。

**环境局限：** 额外在本地torch1.8.0.post3尝试匹配服务器torch1.13.1+cu117的64初始摘要，得到本地`c9b881a28c0b44f993ed299d080a80825c6e8e9a3e5f760709b995a760e7fa2e`与服务器`6adbe5dbf73017ec783a42c920f18236a876b8364f317c8cf5a702c2166b2414`不同，初始化保护正确拒绝。没有将它写成服务器初始化已通过，也没有放宽摘要/版本保护；须由服务器原环境验证。静态合同通过、CPU合成测试通过与真实服务器初始化/效果是三个不同层级。

只取同批8的原ROI，64参考头仅在CPU短暂构造用于初始拷贝；优化器只包含16头参数，没有B图/整张FPN/GPU双头并行。设计未增加图像batch、ROI或特征分辨率，但真实峰值仍看`refiner_cost`，不能承诺进程显存绝对不变。

### 54.4 服务器运行及结果打包（已准备，未执行）

上传前4个运行文件至服务器相同位置；测试可选。既有G/ROI消融来源、旧baseline报告和原ROI缓存保留原位置。输出目录须不存在；若已有本轮结果，先回传已有结果，不覆盖重跑。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_capacity_v1.py \
  --check-only \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --out-dir work_dirs/port_geometry_g_capacity_v1_static
```

确认静态成功后运行固定有限检查（物理卡3/逻辑0；单卡是这项200步检查的设计，不是完整训练）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_capacity_v1.py \
  --gpu 0 \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache \
  --out-dir work_dirs/port_geometry_g_capacity_v1_train
```

成功状态`TRAIN_CAPACITY_CHECK_COMPLETE_REVIEW_REQUIRED`，新增`head_updates_total=200`、`additional_reference_updates=0`、B前向/更新0、无checkpoint导出，初始B与共同初始化摘要核对通过。FAILED时保留并回传结果目录，不跳过环境/初始化/source保护、不加宽度候选或改步数。

结果目录含completion、compact_fc16备份、progress、artifacts、protocol/source副本和cache_manifest；主报告附带历史64参考，后续无需再次单独上传旧报告。没有复制原feature payload、B权重或12张已审查预览。两目录压缩后可只回传一个包：

```bash
bundle="work_dirs/port_geometry_g_capacity_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$bundle" -C work_dirs \
  port_geometry_g_capacity_v1_static \
  port_geometry_g_capacity_v1_train
printf '%s\n' "$bundle"
```

下一次回传按用户偏好在临时目录读取/复核，结论集中追加本文，不另存一次性review脚本或复制整套服务器结果到本地work_dirs。当前只是容量检查入口交付，尚无新几何收益或正式训练许可；TEST已多次暴露，本轮不使用TEST调参或选权。

## 55. FC16容量检查回传：工程通过，联合几何未通过，收束容量尝试（2026-10-03）

### 55.1 回传身份及必要复核（事实）

本轮读取`/Users/mac/Downloads/port_geometry_g_capacity_v1_review_20261003_090732.tar.gz`及终端。包294,667字节，SHA `5862b2a629da237ff50dcc85e58eb1746d74ebcf2c485fa0740bc681c6b960a6`，12个普通文件、无链接/路径越界。直接从压缩包读取JSON，临时依赖缓存用系统临时目录并在退出时清理；未展开到项目目录，没有新增本地复算脚本、报告、压缩包或work_dirs结果副本。只更新本文，未修改运行源码、连接服务器、训练/推理或读取新VAL/TEST。

- 静态/拟合状态分别为`STATIC_CAPACITY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`和`TRAIN_CAPACITY_CHECK_COMPLETE_REVIEW_REQUIRED`。主报告SHA分别`4f7c77fadd178b8104c83d3ddcc228474a5f32100bfe64e311924af8d7e6c177`、`bdd42d4d835bca4ebeebb305453bf80531b822db7919a3e4e835a1303ac3dc5f`。两份artifacts绑定的10个文件及完整清单通过，其余两个文件是artifacts自身；静态progress为空正常。
- 当前59项来源、固定TRAIN标注/64图fixture、协议、baseline、缓存manifest匹配本地固定版；参考FC64完整result与原ordinary报告一致。小头备份、主报告、10条progress快照的result/cost/identity一致；同200个4real+4sim批次与既有参考及seed1703重建结果一致，probe未优化。
- 本次实际新增16头200步、64参考新增0步、B前向/更新0，无checkpoint导出。服务器仍为原torch1.13.1+cu117等环境；物理卡由3改为2，逻辑0与单进程符合合同，不是对照变更或完整训练。
- **服务器初始化缺口已闭合：** 64未训练头摘要精确为历史`6adbe5dbf73017ec783a42c920f18236a876b8364f317c8cf5a702c2166b2414`；16头初始摘要`28667f903a5bcd27f9ffacf3f10d4d4f6312500a436d5cda3d6b7850a0c0fcf5`、59,107参数/无buffer，报告与备份一致。共同stem/固定前16单元/零输出检查通过，新头初始128行结果精确B且与FC64参考初始相同。该结论依据服务器受SHA绑定的执行与报告；没有上传实际初始或最终头tensor，不能称本地独立重读权重。第54节本地跨torch版本摘要不相同保留为历史限制，不是本次服务器失败。
- 保存框复算中心、边长、周期角、分组汇总与两种delta通过，128行GT参考/服务器解析五分量差均0；中心、score、输出数保持，原有界合法框约束通过。RIoU相关复算最大差`4.680155318514956e-6`（GT中心替换增益），独立NumPy原图SmoothL1平均loss复算最大差`3.083010624571614e-8`。保留服务器原值，浮点复核容差不是放松评价门槛。
- 首步三任务梯度norm为.038775/.029486/.023349，均有效，step2 stem norm .000598有效；finite更新检查通过。10条快照裁剪系数均1、记录norm最大1.134355<10，最终尺寸投影0、三个残差饱和计数均0。没有发现解释本次失败的初始化、配对、梯度数值、归一化/分母或坐标错误；未重复未改版的14项CPU测试或完整审计，不冒称10条快照是200步完整裁剪轨迹。

### 55.2 probe逐域逐尺度（事实）

每组8视图，总16个TRAIN图名/32个相关视图，B见过全部图，probe亦已用于多轮开发观察，不能充作独立验证。尺寸为输出框平均绝对相对误差，角度为全部输出的纯pi周期误差；本probe无中心协议角度罚项，fit有1个相关罚项，不能把fit协议角度当纯方向误差。

| probe分组 | 方法 | 长边误差% | 短边误差% | 纯角RMSE° | 纯角p90° | 全帧mean RIoU |
|---|---|---:|---:|---:|---:|---:|
| real原尺度 | B | 3.6494 | 3.6011 | 1.4333 | 2.3732 | .899582 |
| real原尺度 | FC64 | 3.8527 | 3.9133 | 1.3087 | 2.0459 | .897322 |
| real原尺度 | FC16 | 3.8446 | 3.6465 | 1.4835 | 2.4482 | .896985 |
| real半尺度 | B | 4.6231 | 5.2789 | 4.8263 | 7.4295 | .851549 |
| real半尺度 | FC64 | 4.7433 | 4.4352 | 5.0104 | 8.1347 | .850575 |
| real半尺度 | FC16 | 4.7583 | 5.1524 | 4.8077 | 7.3895 | .851919 |
| sim原尺度 | B | 1.8690 | 1.0052 | 1.1984 | 1.8887 | .939004 |
| sim原尺度 | FC64 | 1.8921 | 1.3521 | 1.2266 | 2.1272 | .935460 |
| sim原尺度 | FC16 | 2.0132 | 1.1053 | 1.2309 | 1.7239 | .933495 |
| sim半尺度 | B | 7.8345 | 5.8742 | 1.5627 | 2.1961 | .812432 |
| sim半尺度 | FC64 | 5.3767 | 5.2381 | 1.8919 | 2.5310 | .811843 |
| sim半尺度 | FC16 | 5.8452 | 4.5176 | 1.7360 | 2.1768 | .811344 |

相对B，FC16的real原尺度尺寸/角RMSE/p90/RIoU均退；real半尺度短边、角度及RIoU略好，但长边仍退，RIoU仅+.0003698，不能称稳定或显著收益；sim原尺度尺寸/角RMSE/RIoU退，仅角p90好；sim半尺度两边尺寸与角p90好，角RMSE及RIoU仍退。没有一组达成所有主要几何分量的联合改善，不以p90改善替代RMSE保护。

相对FC64，FC16在real半尺度RIoU提高+.0013438，另3组下降；半尺度角RMSE/p90更好、原尺度角RMSE更差，尺寸取舍混合。32个probe视图相对B RIoU改善11/退化21，严格三个误差同时减小且RIoU升高2/32；相对FC64 RIoU改善18/退化14、严格联合改善5/32，但整体RIoU仍较低。严格计数是描述，不是新放行门槛。

输出覆盖32/32、仅输出中心正确（<15px）32/32、全帧中心正确32/32分别为100%；128稀疏视图三者亦分别128/128。中心逐值复制B，**没有新定位精度收益**；这些覆盖不外推为全视频正确或连续性保持，既有B real VAL仍输出374/375、输出条件正确360/374、全帧正确360/375。原图等比例变换/还原与深度约束保持，深度精度未测。

### 55.3 fit/probe、尾部和容量解释（事实/推断分开）

| 完整角色统计 | B | FC64 | FC16 |
|---|---:|---:|---:|
| fit原图平均loss（96视图） | .1220326 | .0933661 | .1054167 |
| fit长/短边误差% | 3.5252 / 3.4849 | 2.9009 / 2.4560 | 2.9494 / 2.8424 |
| fit纯角RMSE° | 2.9595 | 2.6323 | 2.9417 |
| fit RIoU | .871079 | .877141 | .873840 |
| probe原图平均loss（32视图） | .1390209 | .1300457 | .1330724 |
| probe长/短边误差% | 4.4940 / 3.9399 | 3.9662 / 3.7347 | 4.1153 / 3.6055 |
| probe纯角RMSE° | 2.7030 | 2.8240 | 2.7314 |
| probe RIoU | .875642 | .873800 | .873436 |

FC16四组fit loss均低于B，但均高于FC64；fit角度收益大幅减弱，sim原尺度fit RIoU甚至略低于B（-.0000487）。probe短边/整体角RMSE相对64更好，长边/平均loss/RIoU更差；整体RIoU相对B为-.0022061、相对64为-.0003642。终端末批loss .0808678是同末批优化前loss，不是完整角色均值，初始loss相同为零输出身份的预期现象。

**推断界限：** 缩小FC改变了拟合程度和分量取舍，但没有把fit获益转成联合probe收益。本单点结果不支持“只压缩FC就能解决当前问题”，没有验证容量过大是主要原因；不能排除残余容量/优化问题，也不能反过来据此认定必须加步数、加宽或直接全TRAIN。既有ROI消融相对信息作用、proxy分量目标与固定中心下交叠耦合仍是不同机制，不能混写成同一已证明根因。

退化不只是单个异常框：FC16相对B的sim原尺度RIoU为1改善/7退化，real原尺度2/6。最大负差`sim_seq08_00617`原尺度RIoU .934498→.911973，中心仅.6682px不变，长/短边误差3.5912%/2.1564%→5.1222%/2.9875%，角1.0644°→1.3206°；这说明不能把所有损失都归于中心固定。

固定中心限制仍有具体证据：半尺度`sim_seq08_00168`的中心7.1989px，占GT短边37.33%，虽通过15px命中阈值，定位误差在目标尺度上仍大。FC16将长/短边误差16.8645%/13.7110%降为9.2061%/5.8987%，但角.9777°→1.8174°、RIoU .527094→.516222；比64的.511584稍好，仍低于B。中心“命中”不是中心定位已足够精确，更不等于尺寸/角度与交叠联合正确；该例不证明解冻中心就必然解决全部退化。

本次小头阶段peak allocated3.42725MiB（原64为5.33105MiB）、reserved6MiB，实际没有阶段allocated增加；CPU ROI仍21,316,608字节。耗时约6.58s，卡号/负载不同，不能据此给速度提升结论。统计是PyTorch阶段峰值，不是进程总显存或后续全TRAIN成本。

### 55.4 判断及下一步有限方向（建议，尚未实现）

**保留B ep24；不推荐当前G/FC16正式全TRAIN或新增TEST。** 工程与梯度检查通过，联合几何条件没有通过。结束已预声明的这次容量尝试，不用同probe继续试8/32宽度、选中间step、加步数或放松覆盖/尺寸/角度条件，也不按域尺度拼头。不是宣告所有局部细化路线无效，亦不证明正式全TRAIN一定失败。

下一步建议另行设计**冻结B后的受限中心—尺寸—方向联合细化**，回到用户还未被当前固定中心头覆盖的定位目标。它与D的区别应明确：B backbone/FPN/分类/分配和输出帧决策保持冻结，新增头只修正既有输出几何；D是在原共享检测训练中增加中心补偿，不能把新设计称作已证明的D修复。此建议针对一个已知可优化限制，不宣称它就是FC16失败的主要根因。

拟设计范围：在原单帧ROI条件下增加两个有界中心残差，与现有logL/logS/pi周期角一起受同一GT框的几何监督；保留原固定中心G作机制对照，不新增教师、候选排序或可靠性拒绝。不要同时再改容量、分辨率、ROI采样或增加多种重叠损失；候选头宽度沿用原64作为对照结构，不从本轮小差异选“最佳容量”。中心残差尺度/原图映射、损失归一化及系数需在实施前单独固定；优先用既有fit TRAIN证据明确其可达目标，不用已多次观察的probe/VAL/TEST搜索上限或系数。

**保护要求改变须明示：** 输出帧数量/原分数可继续保持B，中心一旦可动，输出条件中心正确率与全帧中心正确覆盖不再由复制中心保证。必须评价普通real中心mean/RMSE、所有帧覆盖、新增严重错位/RIoU尾部，不能只评价中心loss或删除不利帧。有界中心无法恢复原无输出或修复大范围选错对象，不能保证时序/深度；仍保持等比例与原图坐标/内参约束。

先完成这一新方向的固定设计与必要TRAIN数学/梯度核对，再讨论一次有限对照，当前尚未固定上限/系数、修改源码或启动任何检查。后续正式训练必须另有预算与VAL选权合同并满足原17项条件，不能由本次完成状态自动放行。当前TEST已多次暴露，本轮无任何新TEST/VAL读取、阈值调整或权重重选；可靠性工作保持由另一对话独立推进。

## 56. 受限中心—尺寸—方向细化：固定设计与有限TRAIN检查代码（2026-10-03）

### 56.1 授权、机制与文献边界

用户在了解第55节建议及论文依据后授权实现、复核和服务器指令。本轮新增独立`G-center v1`入口及工具，没有修改原G/FC16、B/D/E-H/F-S、采样或可靠性源码，没有连接服务器或启动真实GPU拟合。工作区其他对话新增的R1.9记录保留。第55.4节“尚未实现”保留为当时状态，最新实施状态以本节为准。

**待验证问题：** 同一冻结B、ordinary ROI、FC64、初始化和200步预算下，允许受限中心修正能否改善定位，并使尺寸/方向收益转化为联合几何收益。当前已知固定中心留下可观的相对定位误差，但中心很准的框亦退化；新增自由度与中心监督一起改变，不是独立证明固定中心就是泛化失败的主因，更不证明D被修复。

借鉴[RoI Transformer（CVPR2019）](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf)的五参数局部回归形式；[R3Det](https://arxiv.org/abs/1908.05612)和[S²A-Net](https://arxiv.org/abs/2008.09397)提醒细化后的框与特征匹配仍有影响，本轮控制变量，保持原ROI而不同时加入这些论文的特征细化/对齐模块。[KLD（NeurIPS2021）](https://proceedings.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html)支持参数耦合的机制分析，项目已用SymKLD，不据论文再叠加KLD/IoU/形状权重。文献已在上一轮读取一手来源；没有论文验证本项目`.30`上限或新增中心项系数。

### 56.2 固定公式、fit依据和对照（设计已实现，效果待验证）

只复用原报告中96个fit TRAIN视图的真实B/GT框，检查所需中心移动距离除以**冻结原图B短边**。各组24视图最大比值分别为real原尺度`.0915349490`、real半尺度`.1858157988`、sim原尺度`.1261981235`、sim半尺度`.2740855355`。未读取probe目标来选择范围，也未用VAL/TEST参数搜索。固定`.30`作为单次保守范围，96个fit目标均在其内部；不声称最优，不因probe失败再放宽。

沿用原257→32→32卷积stem、9×9 flatten＋原四维模型坐标描述符→FC64→3，新增共享FC64后的2维零输出层。参数183,907→184,037，只增130。无绝对中心描述符、BN/dropout或学习采样偏移。原尺寸倍率`.8～1.25`、方向`±10°`及宽高表示/跨边投影完全复用原G；在线不输入GT、图名、域、序列或匹配资格。

令`S_B=min(B原图raw w,h)`、`z_xy`为新增两维原始输出：

\[
 c'=c_B+.30S_B\frac{z_{xy}}{\sqrt{1+\|z_{xy}\|_2^2}},\qquad
 \|c'-c_B\|_2\le .30S_B.
\]

这是二维径向界，不是每轴`.30`而导致总位移`.30sqrt(2)`。实现使用代数等价的数值缩放避免极大有限输出平方溢出，零点Jacobian有效；空输出有旧torch兼容分支。中心在原图xy修正，沿用原等比例变换/还原与原raw尺寸约定；未增加图像边缘裁剪、目标裁剪、按中心拒绝、候选重排或重新取样。原分数/输出数量保持；中心正确率不再由复制保证。

原三个归一化SmoothL1分量`p_L,p_S,p_theta`各系数`1/3`保留；新增`e_xy=(c'-c_GT)/(.30S_B)`、`p_x,p_y=SmoothL1(beta=.1)`，固定：

\[
 L_{joint}=L_G+\frac{p_x+p_y}{3}
 =\frac{p_L+p_S+p_\theta+p_x+p_y}{3}.
\]

**不是五项取平均**，不会无意把原三项监督降为`3/5`；新增项等价`(2/3)*mean(p_x,p_y)`。中心归一化不依赖修正后的尺寸，不能通过放大新尺寸降低该中心项；共享stem/FC仍允许任务取舍，代理分量loss与RIoU仍不等价。所有GT目标保留，不裁剪到残差界、不删除界外视图。

对照：①B零残差；②原ordinary固定中心FC64保存结果，新增更新0；③新joint头200步。严格核验原未训练FC64摘要，完整复制共有张量、只增加零中心层；初始六列精确B。复用原CPU缓存及200个4real＋4sim批次，Adam`.001`、weight decay0、clip10、seed1703，全程仅fit优化，初始/最终200步评价，不挑中间step/seed或按域拼头。B前向/更新0，无训练权重导出，不属于正式全TRAIN。

### 56.3 评价与实现复核（事实/待验证分开）

新入口不复用旧`geometry_deltas`的中心覆盖固定断言，改为逐视图核验输出存在性/score/GT身份，并允许记录中心正确性变化。fit/probe×real/sim×1/.5分别报告中心mean/RMSE/p90与GT短边归一化误差、长短边mean/p90、纯周期角RMSE/p90、严格RIoU mean/p10/min及新增尾部事件。中心正确仍`<15px`，只在输出分母统计；另报输出覆盖和全帧正确覆盖。`>=10px`协议角罚项独立列，不当纯方向误差。

保存逐帧中心错误新增/恢复、RIoU<.5新增/恢复、输出框RIoU恰为0新增/恢复；零RIoU输出沿用历史严重失败口径，与missing分开。对B与原FC64都给有符号差值及全部probe尾部排序。保存“当前中心＋参考形状”和“参考中心＋当前形状”的预测组件替换RIoU，它们只是非线性描述，不是第三/第四个训练消融或可加的因果贡献。

预先固定每个probe组18项不退化检查：中心mean/RMSE/p90及相对误差p90、两边mean/p90、纯角RMSE/p90、输出覆盖、两个中心正确口径、RIoU mean/p10、新增中心错/零RIoU/RIoU<.5保护；另报每组中心mean严格改善及其他几何至少一项严格改善。它们是有限检查规则，不替代后续原17项VAL条件，不自动批准正式训练。32个probe视图被多次用于开发、相互相关、B见过全部；不以微差称显著，不把稀疏点报成全视频连续性。当前TEST多次暴露继续披露，本轮无新VAL/TEST访问或调参。

本地17项必要CPU测试通过（`.59s`）：共有初始化精确、错误/已训练/非有限初始化拒绝，零残差B身份，径向界/等比例单位/原图反射还原/宽高等价，极大中心输出及空输出，冻结B尺度的中心loss与GT/在线输入梯度隔离，原形状loss/共有梯度精确，两步真实CPU优化与有效中心及step2 stem梯度，probe目标不参与范围，缺输出/中心命中三种分母，SHA拒绝、静态入口不加载缓存/头/GPU、完整CPU模拟入口/失败保留与不可覆盖。测试发现并修复torch1.8空张量max路径；同版本已通过的原G/FC16测试未重复执行。

最终真实来源静态入口在系统临时目录运行，通过当前59来源、TRAIN标注/64图/两尺度/固定baseline检查，状态`STATIC_CENTER_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`；fit范围四组各24、界外0，Python3.8 AST通过。输出退出时清理，没有新增本地work_dirs报告、解包副本、一次性review脚本或压缩包。

本地torch1.8与服务器torch1.13初始化摘要不同，拒绝跨环境复用历史初始化是正确行为；本地数值/模拟测试使用同本地版本的合成参考，不冒称服务器复现。服务器仍需在原环境核验原FC64摘要和新头初始B身份、五任务梯度、完整200步clip前后轨迹及真实联合效果。每步新增loss/raw五项/裁剪前后norm/系数/stem norm全部保存，初始有效梯度另报；未执行GPU，不给GPU峰值保证。新头阶段成本会实测，缓存仍21,316,608字节，不加载完整B/FPN图。

### 56.4 文件身份及服务器命令

上传以下四个**新增运行文件到同名项目位置**；测试及本文可随项目同步，不用打代码提交包，不覆盖原G或旧结果：

| 文件 | SHA256 |
|---|---|
| `crane_project/utils/port_geometry_refine_g_center_v1.py` | `790dcba0b1d286b3811f99969b40ac81258353e84fcc67f22621e45e050a8d7b` |
| `crane_project/tools/preflight_port_geometry_g_center_v1.py` | `a77b1fd03af87ad670744e292970d63949cbcf955d61f7bbfe27ea134e4c7453` |
| `crane_project/tools/port_geometry_g_center_v1_protocol.json` | `c7278c004971bf38370fc9ee1010c34836827e185de97e5cec844d4a2fbf2b6c` |
| `crane_project/tools/port_geometry_g_center_v1_sources.json` | `5ff6be9e37e0b3fb51dca2bab55f169853532270326929fd2b4a3c424e6b9ff3` |
| `tests/test_port_geometry_g_center_v1.py` | `58abf1944edf5e5d81a8550a8b99b7c608274d307903b1af0e324f2d642f63aa` |

复用服务器既有原报告和完整缓存，使用原`mmrotljj`环境。先静态：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
conda activate mmrotljj
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_center_v1.py \
  --check-only \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --out-dir work_dirs/port_geometry_g_center_v1_static
```

静态成功后运行有限200步；仍是单卡小头检查，物理2映射逻辑0，原B与固定中心参考均不重训：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_center_v1.py \
  --gpu 0 \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache \
  --out-dir work_dirs/port_geometry_g_center_v1_train
```

预期完成状态`TRAIN_CENTER_CHECK_COMPLETE_REVIEW_REQUIRED`，不是效果通过。目录不允许覆盖；若存在，明确换新out目录，不删除原结果。来源/版本/初始化/缓存不符则先解释差异，不刷新历史SHA或关闭保护。只在两项完成后，服务器压缩回传结果目录（不含缓存、旧模型或提交源码包）：

```bash
tar -czf "work_dirs/port_geometry_g_center_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs port_geometry_g_center_v1_static port_geometry_g_center_v1_train
```

随后按固定评价审查新增中心失败、尺寸/纯角/RIoU联合变化、fit/probe差异及完整梯度轨迹，再讨论是否值得正式训练。仍保留B ep24；不安排新VAL/TEST选权或把有限检查完成当作方法收益。

## 57. G-center v1回传复核：中心可学检查通过，联合probe目标未通过（2026-10-03）

### 57.1 回传身份及工程证据（事实）

读取用户终端和`/Users/mac/Downloads/port_geometry_g_center_v1_review_20261003_100547.tar.gz`，415,703字节、SHA `9a046972ab5c75e7c69c86b67928f2706bd9c4660009b0ab316b3dd928207e0d`。12个普通文件、无链接或路径越界；两个artifacts绑定的10个文件及完整清单逐项通过。静态/main/小头备份报告SHA分别`214bb3bd3b9dc79b0967c253edaa70ea78dccd8be1dd45bdc0718b9e4c4840bd`、`7adb734eae56d703584f2a1a0d025465c07bf80b6ddb18612df0cdb07bf10452`、`8abaef0a72a67f23d87a054259fa425a8ad82d1a6aeae7f0aaf1764131656d26`；progress SHA `22eadbf7055b9556f0b19973d294a92ef3cf502c3b68113d6539eebb1c9939b5`。协议/来源SHA仍精确第56节固定版。

当前59来源、TRAIN标注/64图/两尺度、固定baseline与缓存manifest身份通过；静态/main的identity精确当前重建值。固定中心FC64完整结果仍精确原ordinary，200个批次仍精确原schedule，均只用fit、4real+4sim；所有128行原B/GT保持同一配对。静态状态`STATIC_CENTER_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`，GPU状态`TRAIN_CENTER_CHECK_COMPLETE_REVIEW_REQUIRED`。实际新增头更新200、原FC64/B更新0、B前向0、无checkpoint导出，不是正式全TRAIN。

**服务器初始化缺口闭合：** 原FC64初始摘要仍`6adbe5dbf73017ec783a42c920f18236a876b8364f317c8cf5a702c2166b2414`；新头184,037参数、无buffer，摘要`4a64b65f1e33edf248f7045be7dc919eb1b62713c00348bbb3f7705c8f9a2e97`。共同张量/新增零中心层校验通过，128初始输出与B精确一致，初始原形状loss一致；服务器数值检查确认原形状共有梯度精确不变。初始新增中心两任务加权梯度norm`.0328863/.0033326`，五任务均有效；step1 stem为0符合零末层设计，step2 stem `.00125916`有效。上述依受SHA绑定的服务器代码/报告；实际初始/最终头tensor及ROI payload未回传，不能称本地独立重读权重或特征。

200条progress逐条精确主报告日志，step连续1～200、主/备份结果一致；全部loss/梯度记录有限，裁剪系数均1，最大preclip norm `1.6623712<10`、median `.5061002`、p90 `.9531155`，无裁剪步。clip前后范数复算差最大`1.22e-7`属于数值舍入。最终尺寸投影0，五分量/中心径向饱和计数均0，所有保存框的分数、尺寸范围、中心位移界通过。没有发现能解释此次失败的来源、初始化、丢失梯度、剪裁、坐标或分母错误；同版本已过的CPU测试未重复运行。

保存初始/最终框重算中心、边长、周期角、RIoU、分组、两种对照delta、组件替换及预声明checks通过。RIoU相关跨平台复算最大差`2.5387552e-6`（GT中心替换增益），独立NumPy原图loss/五项最大差`9.7247770e-8`，每步总loss与五项/3差最大`2.7318796e-8`。保留服务器原值；复算使用RIoU相关`1e-5`、其他几何`1e-8`数值容差，不是放宽性能门槛。fit目标支持的跨NumPy浮点摘要按容差核验，不强求小数摘要字典逐位一致。

### 57.2 probe分域/尺度与覆盖（事实）

各组8视图，共16个TRAIN图名/32个相关视图，B见过全部，probe已多轮开发使用，不是独立VAL。尺寸为输出框平均绝对相对误差；角度为全部输出纯pi周期误差。probe均无10px协议角罚项，不把fit罚项变化作为纯方向收益。

| probe分组 | 方法 | 中心mean px | 中心RMSE px | 长边误差% | 短边误差% | 纯角RMSE° | 纯角p90° | 全帧mean RIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| real原尺度 | B | 1.4376 | 1.6700 | 3.6494 | 3.6011 | 1.4333 | 2.3732 | .899582 |
| real原尺度 | 固定中心64 | 1.4376 | 1.6700 | 3.8527 | 3.9133 | 1.3087 | 2.0459 | .897322 |
| real原尺度 | joint | 1.5694 | 1.8150 | 3.7660 | 3.4249 | 1.3232 | 2.3116 | .894569 |
| real半尺度 | B | 2.2026 | 2.5677 | 4.6231 | 5.2789 | 4.8263 | 7.4295 | .851549 |
| real半尺度 | 固定中心64 | 2.2026 | 2.5677 | 4.7433 | 4.4352 | 5.0104 | 8.1347 | .850575 |
| real半尺度 | joint | 2.3941 | 2.6404 | 4.8355 | 4.8740 | 5.2062 | 8.0792 | .842528 |
| sim原尺度 | B | 1.0241 | 1.1597 | 1.8690 | 1.0052 | 1.1984 | 1.8887 | .939004 |
| sim原尺度 | 固定中心64 | 1.0241 | 1.1597 | 1.8921 | 1.3521 | 1.2266 | 2.1272 | .935460 |
| sim原尺度 | joint | 1.0117 | 1.1397 | 1.8552 | .9985 | 1.2592 | 2.0888 | .934791 |
| sim半尺度 | B | 2.9398 | 3.5030 | 7.8345 | 5.8742 | 1.5627 | 2.1961 | .812432 |
| sim半尺度 | 固定中心64 | 2.9398 | 3.5030 | 5.3767 | 5.2381 | 1.8919 | 2.5310 | .811843 |
| sim半尺度 | joint | 2.6888 | 3.0981 | 5.5950 | 4.8178 | 1.6775 | 2.2172 | .816238 |

相对B：real原尺度中心mean/RMSE及长边退，短边/角度好，RIoU-.0050133；real半尺度中心mean/RMSE、长边、角度退，短边好，RIoU-.0090216；sim原尺度中心mean/RMSE和尺寸微好，但角度及RIoU退（-.0042131）；sim半尺度中心/尺寸/RIoU好（+.0038052），角RMSE/p90仍退。预声明每组18项条件分别失败7/9/4/2项（顺序real原/半、sim原/半），没有一组全部通过，不因局部或微小收益放宽条件。相对固定中心64，RIoU亦三组退、一组好，半尺度sim+.0043948不能替代对B的完整目标。

32个probe视图相对B中心变好15/变差17，RIoU变好11/变差21；“两边绝对误差及纯角误差同时下降＋RIoU上升”仅1/32，作为描述不是新放行标准。输出覆盖32/32、仅输出中心正确(<15px)32/32、全帧中心正确32/32分别100%；128稀疏视图亦分别128/128。所有组新增/恢复中心错误、RIoU<.5或零RIoU输出事件均0，但中心距离已经变化，不能再称复制中心；稀疏全命中不外推全视频real覆盖/连续性。原B real VAL仍374/375输出、360/374条件正确、360/375全帧正确。等比例/原图/深度接口约束保持，深度精度未测。

### 57.3 拟合、中心偏置及组件作用（事实与推断）

| 汇总 | B | 固定中心64 | joint |
|---|---:|---:|---:|
| fit中心mean/RMSE px | 2.3584 / 2.9724 | 2.3584 / 2.9724 | 2.2161 / 2.7764 |
| fit长/短边误差% | 3.5252 / 3.4849 | 2.9009 / 2.4560 | 2.9150 / 2.6898 |
| fit纯角RMSE° | 2.9595 | 2.6323 | 2.5485 |
| fit RIoU | .871079 | .877141 | .881583 |
| probe中心mean/RMSE px | 1.9010 / 2.3978 | 1.9010 / 2.3978 | 1.9160 / 2.3002 |
| probe长/短边误差% | 4.4940 / 3.9399 | 3.9662 / 3.7347 | 4.0129 / 3.5288 |
| probe纯角RMSE° | 2.7030 | 2.8240 | 2.8834 |
| probe RIoU | .875642 | .873800 | .872031 |

joint整体fit比B更好，probe整体RIoU相对B-.0036107，相对64-.0017688，角度RMSE更差。joint总loss：fit `.1866098→.1533653`、probe `.1946656→.1840659`，但probe中心项`.0556447→.0557583`微增，probe总loss下降来自形状项`.1390209→.1283076`；不能拿总loss下降证明中心监督有效。终端step200 `.1382381=.0799690+.0582691`为固定末批优化前数值，批次不同，不是全fit均值，也不能和原三项总loss直接比较。fit仅1个10px协议角罚项被移出，纯角改善需看单列。

real原尺度fit中心mean `1.6174→1.6551`、中心loss `.0334386→.0338963`亦退，24视图中心改善8/退化16；该现象不只是“fit学好、probe过拟合”。real原尺度预测中心平均位移fit约(-.3779,+.1746)px、probe(-.3957,+.1993)px（图像y向下）；fit23/24向左、22/24向下，probe8/8向左下，而GT所需向左仅fit9/24、probe3/8。归一化位移均值fit(-.011995,.006139)、probe(-.012354,.006511)，所需目标均值分别(.002809,.008369)/(.014866,.003514)。这是当前稀疏样本的系统性方向偏置事实，不能外推所有real帧，也不能据均值单独确认捷径或梯度冲突。

实际所有128视图最大移动仅`.0739286*S_B`，远低于`.30`界，无饱和，不支持“范围太小导致本次real失败”；不据此扩大上限。fit中心项三组下降、一组退，probe中心项只有sim半尺度下降、其余三组上升；归一化SmoothL1和中心欧氏mean/RMSE本来不是同一统计。

复用保存的预测组件替换RIoU（无需新推理/拟合）：

| probe | 原B | 学到的中心＋原B形状 | 原B中心＋学到的形状 | joint |
|---|---:|---:|---:|---:|
| real原尺度 | .899582 | .896588 | .899546 | .894569 |
| real半尺度 | .851549 | .845440 | .848919 | .842528 |
| sim原尺度 | .939004 | .935734 | .937951 | .934791 |
| sim半尺度 | .812432 | .814901 | .813803 | .816238 |

real两组及sim原尺度单独使用本次学到的中心也未提高平均RIoU；所以“只撤掉新形状预测、保留该joint学出的中心”仍不能解决。表中各项非线性、中心参数在联合训练中学得，不是独立中心单任务对照，不能把差值相加当贡献，也不能凭它证明独立中心训练必然失败。sim原尺度中心欧氏mean/RMSE微好但中心替换RIoU退，说明优化的距离统计与交叠并非一一对应；根因仍未确定。

正例`sim_seq08_00168`半尺度：中心7.1989→6.1701px，长/短边误差16.8645%/13.7110%→8.1828%/4.4875%，RIoU .527094→.547065（原64 .511584），角.9777°→1.8365°仍退。仅新中心＋B形状RIoU .547947，仅新形状＋B中心 .514500；该例支持中心确实能局部解除交叠限制，不能充当全体收益。负例`real_seq05_00100`半尺度：中心1.3290→1.9792px，RIoU .850282→.782390；`real_seq05_00420`原尺度中心.9897→1.2144px、角1.2630°→2.2150°、RIoU .863494→.848337。不删除这些普通帧。

**推断：** 工程/梯度路径已确认有效，零初始化形状强度保持，无剪裁压制、残差饱和或丢梯度证据。当前需要解释的是样本间纠偏规律和任务/目标与交叠的关系。共享stem/FC上的中心—形状梯度干扰、冻结特征定位支持不足、固定200步学习动态/统计偏置和代理目标不一致均仍是待验证解释；没有记录各步共享任务梯度夹角，不能由正梯度norm或偏置确认哪个主导。不能证明“解冻中心解决所有问题”，也不能宣告全部局部细化方法无效。

服务器GTX1080/原torch1.13.1+cu117环境：小头检查耗时约8.34s，peak allocated5.99365MiB、reserved28MiB，缓存CPU21,316,608字节。reserved较原固定中心阶段6MiB增加，allocated较5.33105MiB增加约.66MiB；本次未发生大幅allocated增长，但reserved也必须披露。测量含初始配对梯度检查和拟合，是阶段allocator数据，不是进程总显存、正式训练上限或严格速度对照。

### 57.4 判断和下一步建议（尚未实施）

**保留B ep24；不推荐G-center v1正式训练、新VAL/TEST、扩大位移界、扫中心系数/容量/中间step或直接延长200步。** 工程通过与方法达到目标必须分开；sim半尺度局部正例和稀疏全中心命中不能覆盖real定位及sim角度退化。当前TEST已多次暴露，本轮无新TEST/VAL读取、调参或权重重选。

若继续冻结ROI路线，只建议下一项固定的**中心单任务机制对照**作为收束：保留相同ordinary ROI、FC64共有初始化、中心径向`.30*S_B`/归一化/每轴`1/3`、Adam/seed/同200批次；尺寸/角度精确B，只优化两个中心分量及对应共有stem/FC。比较B、当前joint“预测中心＋B形状”保存框、新独立中心头；不同时改采样/分辨率/容量/权重/步数，不把三项常数loss计入学习收益。目的仅是检验移除尺寸/角度梯度后中心纠偏能否改善，而非直接提出最终五参数方案。

仍完整报告原三个覆盖分母、原图中心mean/RMSE/p90、RIoU/尾部及真实修正方向；大小/角度精确B不是新的精度收益，稀疏中心不保证连续性/深度。若独立中心仍不能保持real定位并改善交叠，则停止本轮同缓存P3＋FC64残差头的增量试验，再单独设计更有代表性的源域几何支持或特征精度证据，不能继续用同probe搜索“成功参数”。若中心单任务更好，也只能支持该固定优化设置下的任务交互影响，不能单独证明共享冲突是整个项目根因，更不能自动组合成最终方案。上述代码未改、检查未启动，需后续授权。

本轮仅更新本文：直接读取压缩包、临时依赖缓存退出清理，不留存解包副本、review脚本或新结果文件。没有修改原运行源码、重跑未变化的测试、连接服务器；保留另一对话新增的可靠性诊断文件。

## 58. 下一项中心单任务对照及针对性文献（2026-10-03，设计未实施）

用户本轮请求具体建议和论文搜索，未请求实施。沿用57.4的有限对照，不启动正式训练或改运行代码。

### 58.1 要回答的问题与固定变量

事实依据：joint的real原尺度fit中心也退化，不能只归为probe过拟合；real纠偏出现近一致左下偏移；无剪裁/饱和证据；保存的“joint中心＋B形状”在三个probe组RIoU仍低于B。待验证问题是：同一输入、初始化和预算下，去掉形状监督对共有表示的更新，是否能改善中心纠偏？这不是已经确认梯度冲突，也不是最终尺寸/角度方案。

建议名称G-center-only v1。复用ordinary P3 ROI、描述符、stem/FC64及共同初始化，两个中心输出零初始化；可保留原三形状输出层但冻结且完全旁路，避免顺带改变随机初始化。输出w/h/angle及score/输出帧精确复制B。中心解码仍`c'=c_B+.30*S_B*z/sqrt(1+||z||²)`，`S_B`为detach原B短边。唯一优化loss为`(SmoothL1(e_x,beta=.1)+SmoothL1(e_y,beta=.1))/3`，`e=(c'-c_GT)/(.30*S_B)`；保留原每轴1/3，不改为两项均值，不把常数形状误差计为训练收益。

固定seed1703、Adam lr.001/wd0、clip10、batch8(4real+4sim)、同200批次；48个fit图名/96视图用于更新，16个probe图名/32视图仅评价。新增200个小头更新；B前向/更新0，既有joint参考更新0，不导出用于部署的小头权重。比较三臂：原B、保存joint中心＋B形状、新独立中心＋B形状。既有joint完整输出仅作背景，不能拿其形状变化混入中心任务对照。

### 58.2 预先约定的判读与退出

完整报告fit/probe×real/sim×1/.5：中心mean/RMSE/p90、归一化误差及x/y有符号残差/实际与所需位移、严格RIoU mean/p10/min、逐帧好坏计数和paired尾部事件；输出覆盖、仅输出中心<15px命中率、全帧正确覆盖分别报分子分母。尺寸/纯角须精确B，属于实现约束，不计为改善。所有200步loss/norm/clip和最终径向饱和记录保留，不按中间step选结果。

继续研究门槛：四个probe组相对B中心mean/RMSE/p90不退、RIoU mean/p10/min不退，无新增中心失败、低于.5或零RIoU事件；real两个组fit中心mean/RMSE不退。至少一个real probe组中心mean和mean RIoU同时严格改善，且对应指标优于保存的joint中心参考，才说明该对照对当前real问题有继续价值。仅浮点容差范围内变化按持平，微小单次收益不称稳定或显著。以上是下一项机制检查的门槛，不替代原正式VAL条件；不要求不变的尺寸/角度产生虚假严格收益。

若只优于joint仍劣于B：最多说明去掉形状训练减轻了当前设置的损害，不能替换B。若中心距离改善而RIoU退：代理目标/纠偏方向仍未满足需求。若real仍退：停止本轮同缓存P3+FC64增量搜索，不扫权重、宽度、位移上限或步数。下一阶段若需要，另设计代表性TRAIN与定位特征支持检查；不得直接宣布P3是根因。即使通过也只是单次有限TRAIN证据，不能自动正式训练或组合为五参数新方法。

### 58.3 检索核实的论文及借鉴边界

- Yu等，**Gradient Surgery for Multi-Task Learning**，NeurIPS2020：[官方论文页](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)，[作者摘要](https://arxiv.org/abs/2001.06782)。研究多任务梯度干扰及PCGrad投影。借鉴独立任务参照与共享梯度诊断；当前仅有norm没有夹角，不能直接套PCGrad。单任务胜出也只说明任务交互，不证明负梯度内积是主因。若以后测夹角需非零共享梯度且同一参数状态/批次；零末层初始共享梯度为0不能定义有效夹角。当前未保存最终头tensor，旧JSON无法补算，重放joint会额外增加200步，暂不隐含加入本次预算。
- Song等，**Revisiting the Sibling Head in Object Detector**，CVPR2020：[作者论文页](https://arxiv.org/abs/2003.07540)。TSD讨论分类/定位空间需求不同并解耦特征采样。可借鉴“不同任务是否适合共享表示”的问题，但其研究对象是分类与定位，不是中心与尺寸/方向；不是本项目冲突的直接证明，本轮不同时改ROI采样。
- Yang等，**Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence**，NeurIPS2021：[官方论文页](https://proceedings.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html)。讨论旋转框参数耦合、随对象形状调整梯度及尺度不变性。用于解释需联合检查中心/形状与交叠，不能只看欧氏距离；B已用SymKLD，论文不支持在本项目直接再加KLD或调整系数。
- Wang等，**Side-Aware Boundary Localization for More Precise Object Detection**，ECCV2020：[官方论文PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490392.pdf)，[作者摘要](https://arxiv.org/abs/1912.04260)。逐边定位及分桶后细化，提供边界空间证据与定位表示的后续方向。其水平框设置不能直接搬成OBB方案；当前也未证明边界特征不足，因此列为中心单任务失败后的备选研究依据，不在下一实验叠加分桶/边界头。

以上检索支持问题与对照思路，不构成当前方案有效性的证据。本轮未声称穷尽最新文献；没有修改运行代码、训练或连接服务器。当前probe被反复开发使用且B见过，TEST亦已多次暴露；本次设计不用TEST调参或选权。保持等比例、原图还原及深度接口约束不等于验证深度精度或连续性。仍保留B ep24，结论只集中记入本文。

## 59. G-center-only v1实现、代码复核及服务器命令（2026-10-03）

用户授权按第58节修改代码。本节为实施与本地验证事实；不把第58节历史设计阶段或有限检查误写成正式训练/收益。

### 59.1 实现范围及固定契约

新增运行文件（上传时保持项目相对位置）：

| 文件 | SHA256 |
|---|---|
| `crane_project/utils/port_geometry_refine_g_center_only_v1.py` | `4abf4fd9f39c417d2e9035a048020843e590918f4d0935af67ae17e262dadd01` |
| `crane_project/tools/preflight_port_geometry_g_center_only_v1.py` | `80ab470c1acee5f3e1b8356c6fd874c302e9fc6e78b970cc3a9eaf5c1d77a31d` |
| `crane_project/tools/port_geometry_g_center_only_v1_protocol.json` | `8821263cf4787d469b8305c59418c36b5e83e60124c7fb7415a9323cf293ec7b` |
| `crane_project/tools/port_geometry_g_center_only_v1_sources.json` | `8049731476d380866e8969448c0fb54d18787340e542ff793bab34e5b72f9017` |

测试`tests/test_port_geometry_g_center_only_v1.py` SHA `11728a7032334d95562e9581a7fac6708485e3b348e1db66b906f015226fbacb`。旧G、ROI消融、FC16及joint运行源码/协议/manifest均未改；可靠性文件不在本轮修改范围。不生成上传源码包。

新头继承相同stem/FC64及两个中心零输出单元，复制整个已审查joint未训练state。保留195个原形状末层参数以保持state身份，但`requires_grad=False`且前向完全不调用；总参数184,037、可训练183,842。前向仍只接受detach ROI/support、B original/model框描述符，没有GT/域/图名输入；原B raw宽高/角度/score精确复制，仅原图xy变化。稳定径向squash、`.30*S_B`及中心loss两项各`1/3`与joint一致，不能改成两项mean。GT形状与修正后的边长均不能降低中心loss。空输出保留空框接口；训练loss要求非空配对目标。完整来源保护仍要求现有128视图全eligible，不暗中增加/删除样本。

runner绑定旧59来源及新util/runner/protocol共62来源；继续验证原TRAIN、原baseline、原CPU缓存及旧joint身份。旧joint完成报告SHA精确`7adb734eae56d703584f2a1a0d025465c07bf80b6ddb18612df0cdb07bf10452`，由已保存中心＋B形状重算hybrid，无额外joint拟合。实际新初始化仍要求与服务器旧joint完整state digest一致；优化前中心两分量及共有参数梯度须精确旧joint中心任务，冻结形状梯度为None。首步共有stem梯度0符合零末层，step2及全部200步有效性/裁剪另报。

相同seed1703、Adam .001/wd0、clip10、batch8及同200个fit-only批次；新增200更新，joint/B更新0、B前向0，无检测器构建、checkpoint导出或VAL/TEST访问。每步中心loss/原始两项/权重/梯度范数/clip/stem记录保留；初始/最终全部128视图按8组报告。新增逐视图所需/实际位移、有符号残差、符号计数、中心与RIoU好坏计数。覆盖3种分子分母显式列出；纯角不变与10px协议角罚项变化分别解释。

评价严格按58.2实现，RIoU增加min非退化检查；不要求复制的尺寸/方向产生严格收益。预声明数值容差中心`1e-6 px`、RIoU`1e-5`，容差内按持平，严格收益需超出容差；中心命中/低交叠事件及分子分母无容差。`diagnostic_continuation_conditions_met`只表示有限对照符合继续研究条件，`formal_training_approved`固定False，不能自动升级训练。输出目录不覆盖；完成小头结果先保存`center_only.json`，即使后续比较失败也保留200步结果和失败artifacts，不导出权重。

### 59.2 本地复核与剩余缺口（事实）

最终13项CPU测试通过（Python3.8/torch1.8，约1.15s）：共有初始化/可训练划分、形状层彻底旁路、中心两项旧权重、GT形状/尺寸膨胀无影响、旧joint中心梯度精确一致、两步真实优化到stem、输入/GT detach、原raw形状/score保持、极大有限激活/径向界/等比例单位/空框、保存hybrid不修改原报告、缺输出与新增中心失败的三个分母、完整分组/real fit/超容差联合收益门槛、拒绝未审查joint、静态不加载缓存/GPU/head、CPU两步模拟完整入口、失败保留小头结果/原更新预算、序列化失败摘要等。AST按Python3.8解析及JSON解析通过。

真实静态入口在系统临时目录运行：直接读取用户回传tar中joint completion（只在临时目录落盘），结合现有原G baseline，通过62来源及TRAIN/协议/报告身份校验，恢复128个hybrid行，状态`STATIC_CENTER_ONLY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`、更新0。临时输出/依赖缓存自动清理；不在项目新增解包结果、review脚本或复算报告。原同版本测试不重复全量运行。

本地没有服务器实际ROI payload及CUDA环境，尚未验证服务器同初始化digest、真实缓存上的200步、几何效果和显存峰值。两步合成CPU入口是工程测试，不是本项目性能证据。内存结构仍是原CPU小ROI＋小头，不加载B/FPN计算图；参数不增加，但不能承诺特定设备allocator峰值，GPU报告继续记录allocated/reserved及耗时。服务器环境/缓存/初始化不符必须解释差异，不刷新旧SHA来绕过保护。

### 59.3 服务器执行与结果回传

在`/media/omnisky/personal_files/ljj/symEOOD`、原`mmrotljj`环境，上传59.1四个运行文件，保留旧源码/缓存/结果。先静态（使用原baseline和上一轮joint的真实服务器文件）：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_center_only_v1.py \
  --check-only \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --joint-report work_dirs/port_geometry_g_center_v1_train/completion.json \
  --out-dir work_dirs/port_geometry_g_center_only_v1_static
```

静态完成后执行固定有限TRAIN对照（单卡是小头检查，不是完整检测器训练）：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_g_center_only_v1.py \
  --gpu 0 \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --joint-report work_dirs/port_geometry_g_center_v1_train/completion.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache \
  --out-dir work_dirs/port_geometry_g_center_only_v1_train
```

预期完成`TRAIN_CENTER_ONLY_CHECK_COMPLETE_REVIEW_REQUIRED`，不表示精度通过。若out目录存在，换新的目录名且同步下方tar参数；不覆盖或删除旧结果。

```bash
tar -czf "work_dirs/port_geometry_g_center_only_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs port_geometry_g_center_only_v1_static port_geometry_g_center_only_v1_train
```

只回传这两个结果目录，不含ROI缓存、模型权重或源码包。审查新/旧中心纠偏方向、real fit和probe中心/RIoU联合变化后再决定后续；失败按58.2收束同缓存路线。保留B ep24；当前TEST已多次暴露，本轮不用于调参、方法或权重选择。等比例/原图/深度接口约束保持，仍未验证视频连续性或深度精度。

## 60. G-center-only v1回传复核：fit改善，probe三组退化，结束本轮同缓存增量路线（2026-10-03）

### 60.1 身份、工程及复算（事实）

读取终端和用户`/Users/mac/Downloads/port_geometry_g_center_only_v1_review_20261003_110105.tar.gz`，440,621字节，SHA `71031b1eb409132331550da4dac057157b6bec906e932c0979d94379ae335f4a`。12个普通文件＋2个目录，无链接/越界路径；两份artifacts绑定的10个文件完整清单与SHA通过。静态完成报告SHA `fa2be754e540d65dc91a763d40cb45ee9164bab9aee98bb259c498c65e6d4048`，TRAIN完成报告SHA `5f7c70f1f3e65147d9d568fbbe9da23591bf7bd2d83ef68a6b59ad5bdda6b081`；小头备份SHA `2c5896e46223fe9b41f26adc9fe536bf8aacde6bc51c0d58d4c7d370a6c6788a`，progress SHA `64d4bb924a3a130eb5aaa6e7803bc63bc1a4c6720e8cf60aa8825bce1cda5ef9`。协议/manifest仍59.1固定版；当前62来源、TRAIN/原baseline/旧joint、cache manifest与重建identity通过，保存hybrid与旧joint中心/B形状逐视图一致。

静态`STATIC_CENTER_ONLY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`、GPU `TRAIN_CENTER_ONLY_CHECK_COMPLETE_REVIEW_REQUIRED`。静态更新0；GPU新增头200更新，joint/B更新0、B前向0、单逻辑卡0（物理2）、无正式训练/权重导出。200条progress精确主报告日志，连续1～200；小头备份与主报告精确一致；200个批次精确原B/joint schedule，每批8、4real+4sim且均fit，没有按中间step选结果。

服务器初始化与旧joint完整digest精确`4a64b65f1e33edf248f7045be7dc919eb1b62713c00348bbb3f7705c8f9a2e97`；总184,037/可训练183,842，冻结195个形状末层参数。初始128框精确B、中心loss精确旧joint；服务器报告确认共有中心梯度精确一致，两轴加权末层norm `.032886309/.003332602`有效，形状梯度不存在。step1 stem0符合零末层，step2 `.0004641115`有效；全部loss/梯度记录有限、冻结形状state未变、200步均未触发clip，preclip最大`.92900294<10`，median`.22641725`、p90`.50102296`。clip前后复算舍入差最大`5.25e-8`。最终两轴/径向饱和0；保存框最大移动`.10083142*S_B`，远低于`.30`。无丢梯度/裁剪压制/幅度饱和证据，不支持扩大位移上限。

初始/最终全部128行身份、原B raw宽高/角度/score精确保持。独立保存框复算中心、周期角、RIoU/尾部、三个分母、分组、两种paired比较、方向摘要及预声明门槛通过。RIoU相关最大跨平台差`3.7182353e-6`（保存hybrid的GT角替换增益），NumPy中心loss/原始两项复算最大差`1.9555995e-8`，逐步loss与两项/3差最大`9.9341075e-9`；分别保留RIoU`1e-5`、其他几何摘要`1e-8`复算容差。浮点摘要不强求字典逐位相等；不修改服务器值或放宽性能门槛。未发现可解释退化的代码/来源/初始化/分母错误。实际头tensor与ROI payload未回传，梯度与tensor身份依受SHA绑定的服务器代码/报告，不能称本地独立重读权重或特征；同版本测试未重复运行。

GTX1080、原torch1.13.1+cu117环境；检查阶段耗时约4.76s，peak allocated5.98975MiB、reserved28MiB，CPU缓存21,316,608字节。与joint相比allocated基本相同、reserved相同；没有本次显存突增。它是小头检查阶段allocator数据，不是进程总显存或正式训练显存/严格速度对照。

### 60.2 probe结果与覆盖（事实）

每组8视图，共16个TRAIN图名/32个相关视图，B见过全部且probe已反复用于开发；不是独立VAL。旧joint中心列只用其预测xy＋原B raw形状，不混入joint尺寸/方向修正。

| probe分组 | 方法 | 中心mean px | 中心RMSE px | 中心p90 px | mean RIoU | RIoU p10 | RIoU min |
|---|---|---:|---:|---:|---:|---:|---:|
| real原尺度 | B | 1.437614 | 1.670033 | 2.740814 | .899582 | .854898 | .848085 |
| real原尺度 | 旧joint中心 | 1.569428 | 1.814999 | 2.911811 | .896588 | .851559 | .848085 |
| real原尺度 | 独立中心 | 1.601908 | 1.832118 | 2.968926 | .895451 | .859148 | .848085 |
| real半尺度 | B | 2.202637 | 2.567725 | 3.882087 | .851549 | .784250 | .771157 |
| real半尺度 | 旧joint中心 | 2.394128 | 2.640406 | 3.862209 | .845440 | .787183 | .775906 |
| real半尺度 | 独立中心 | 2.371466 | 2.936144 | 4.204747 | .847346 | .762602 | .718346 |
| sim原尺度 | B | 1.024071 | 1.159658 | 1.529572 | .939004 | .918189 | .914072 |
| sim原尺度 | 旧joint中心 | 1.011696 | 1.139665 | 1.525304 | .935734 | .922779 | .916275 |
| sim原尺度 | 独立中心 | 1.181230 | 1.318447 | 1.755906 | .928593 | .907082 | .905864 |
| sim半尺度 | B | 2.939849 | 3.503039 | 4.963766 | .812432 | .687105 | .527094 |
| sim半尺度 | 旧joint中心 | 2.688834 | 3.098091 | 4.536807 | .814901 | .722407 | .547947 |
| sim半尺度 | 独立中心 | 2.738618 | 3.299915 | 4.771325 | .831290 | .749976 | .610117 |

相对B：real原尺度中心三项与mean RIoU退，p10好、min持平；real半尺度及sim原尺度中心/RIoU六项均退；sim半尺度六项均好。预声明12项probe条件失败分别4/6/6/0（real原/半、sim原/半）；仅sim半尺度全部通过。real fit mean/RMSE两组均通过，但real probe联合收益组为空；`diagnostic_continuation_conditions_met=False`，`formal_training_approved=False`。相对旧joint中心亦不一致：real原尺度更差，real半尺度mean/RIoU稍好但RMSE/p90及RIoU尾部更差，sim原尺度更差，sim半尺度RIoU好但中心三项不如旧joint。

32个probe中心改善10/退化22，RIoU改善15/退化16/容差内持平1；不能由汇总平均近持平掩盖三个域/尺度组退化。probe输出覆盖32/32、仅输出中心<15px正确32/32、全帧正确32/32分别100%；全部稀疏视图对应128/128。新增中心失败、低于.5交叠或零RIoU输出事件全0，但该粗阈值不代表定位精度保持。B real VAL仍374/375输出、360/374输出条件正确、360/375全帧正确，稀疏100%不能外推全视频。尺寸/纯角精确B，不是新增收益；本次不能解决尺寸及sim角度目标。

### 60.3 fit/probe差异、偏置和代理目标（事实与推断）

| 汇总 | B | 旧joint中心＋B形状 | 独立中心＋B形状 |
|---|---:|---:|---:|
| fit中心mean/RMSE px | 2.358439 / 2.972354 | 2.216097 / 2.776444 | 1.767997 / 2.398734 |
| fit mean RIoU | .871079 | .875072 | .889791 |
| fit中心loss | .06457717 | .05890405 | .04464347 |
| probe中心mean/RMSE px | 1.901043 / 2.397828 | 1.916021 / 2.300161 | 1.973306 / 2.480188 |
| probe mean RIoU | .875642 | .873165 | .875670 |
| probe中心loss | .05564469 | .05575826 | .05508198 |

独立中心四组fit的中心mean/RMSE、中心loss和平均RIoU均比B、旧joint中心好；fit有71/96中心变好、25/96变差。real原尺度fit mean `1.617447→1.329608`（旧joint `1.655149`），RIoU `.890848→.903131`；原joint的real fit中心退化在单任务设置中被解除。这支持**任务组合改变了固定预算下的拟合动态**，不等于证明共享梯度冲突。它同时说明当前框架可以学到部分纠偏，不能说中心支路无效或ROI完全无信息。

probe中心mean/RMSE却退，三个组仍失败。整体mean RIoU只增`.000027862`，由sim半尺度改善抵消其他三组下降，不称稳定/显著收益。probe中心loss略降约1%，像素中心mean反而增约3.8%；real半尺度loss `.05405114→.05463552`微退，mean/RMSE/p90及RIoU尾部仍明显退。loss按每帧原B短边归一化，像素距离、平方误差、方向性交叠及分组平均是不同目标；聚合loss降不能替代几何指标。这是机制/统计目标的区别，不是损失算错的证据，也不支持立即更改归一化/权重。

real原尺度probe预测平均移动(-.406574,+.112228)px，GT所需均值(+.695728,+.245003)px；7/8向左、实际需要向左3/8，中心改善1/8、退化7/8。旧joint为8/8向左下；独立中心的y方向已有变化，不能继续称全部“左下偏置消除失败”，更准确是**向左纠偏倾向仍在**。real半尺度probe8/8向左、需要5/8向左；平均实际(-.992043,-.006187)px、所需(-.183266,+.590557)px。real原尺度fit亦20/24向左但只9/24需要向左，同时16/24中心改善，说明仅均值/符号不能证明错误捷径或优化主因，仍须结合逐帧残差/短边归一化和样本内容。

正例`sim_seq08_00168`半尺度：中心7.198864→6.611414px、RIoU .527094→.610117（旧joint中心hybrid .547947），尺寸/角度未变。负例`real_seq13_00168`半尺度：中心4.950678→6.075845px、RIoU .771157→.718346；`real_seq05_00100`原尺度中心.459485→1.014075、RIoU .925177→.896646。fit也非逐帧全好：`sim_seq08_00202`半尺度中心4.289423→5.264565、RIoU .718283→.656426，使fit sim半尺度min低于B的.709785；不能简化成全部fit学好、仅probe问题。

复用保存GT中心替换参考：fit mean RIoU `.917926`、probe `.910022`，替换前后保持同值，符合形状未变。该描述性标签替换有改进空间，不说明现有特征可实现它，不当真实模型效果或损失/方案选择证据。

**推断与未验证解释：** 新增形状监督不是本次中心退化的必要条件（移除后仍出现），但不排除旧joint存在任务干扰；本次没有训练轨迹共享梯度夹角，不能宣称PCGrad已被证明适用或无用。当前有限设置的样本纠偏规律未稳健迁移；冻结P3的定位支持、归一化代理目标、序列/外观样本支持、GT边界一致性及200步学习动态仍未拆开验证。全量TRAIN正式训练未发生，不能声称一切局部细化网络都无效。

### 60.4 下一步：结束同缓存头搜索，有限核对几何监督与空间证据（建议，未实施）

**保留B ep24；不推荐本版正式训练或新增VAL/TEST。** 按58.2预先声明，停止本轮同缓存P3+FC64的容量/系数/位移上限/步数增量搜索。当前结果有用之处是完成机制对照、排除“只去掉形状监督就能解决probe定位”的解释，而不是形成最终优化。

下一阶段建议先做一项**TRAIN几何监督—定位空间证据核对**，不再立即训练另一头：

1. 优先复用现有64图/128视图、原B和G系列保存框，分域/尺度看目标有符号中心修正、短边归一化、目标长宽比、中心原本已准/需要纠偏与修正好坏，不改变loss或删样本。先说明当前失败是否集中在普通高精度框、某序列/边界条件，避免把半尺度sim局部正例当通用规律。
2. 补有限原图证据：按预先规则抽取少量TRAIN正例/负例/普通对照，等比例放大显示原GT多边形、B与细化框及中心，核对标注包围对象范围、边界/遮挡和亚像素纠偏是否有可见依据。使用开发probe负例仅作诊断，不当新独立验证，不修改标签来适应模型；若确认标签问题，另立修订与版本对照。
3. 只有前两项仍无法解释时，补核对已保存ROI/P3对真实边界/中心的空间支持，不默认升级网络。原baseline支持报告可直接复用：原尺度四组GT短边median约4.07～4.44个P3单元、半尺度约2.03～2.22，ordinary mask mean均1；mask完整仅说明未越有效图像，不证明边界定位信息充足。9×9插值采样不增加底层独立空间分辨率。sim半尺度本轮反而改善，故不能据2个单元直接把P3分辨率定为唯一根因。

以上只补当前缺口，具体样本数/抽样清单/输出协议应在实施前固定；不重开全项目审计或TEST选型。如果核对支持边界定位证据不足，再把边界空间取证作为一项新的有限设计，而非继续叠加损失或把多个网络变更合并。

本轮重新核实[PCGrad官方论文页](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)与[SABL作者论文](https://arxiv.org/abs/1912.04260)：PCGrad研究任务梯度干扰，现有结果尚不足以直接使用；SABL逐边定位/分桶细化可作为空间边界证据不足时的后续参考，但其水平框方案不直接等于本项目OBB优化，也没有验证本项目收益。

本轮只更新本文：回传tar直接读取，系统临时目录退出清理，没有项目内解包/复算副本、review脚本或新结果文件；没有修改运行代码、重跑未变化测试、连接服务器。TEST已多次暴露，本轮无新TEST/VAL访问、调参或选权。等比例/原图/深度接口约束保持；稀疏输出100%不保证视频连续性和深度精度。可靠性仍由另一对话推进。

## 61. 有限TRAIN几何监督与像素定位证据核对：实现、复核及服务器指令（2026-10-03）

### 61.1 实现范围与来源（事实）

按用户授权实施60.4的前两项，第三项真实ROI/P3响应核对仍未实施。新增CPU只读入口，不构建检测器/细化头、不读ROI payload、不更新任何参数、不导出权重、不访问VAL/TEST、不修改标签；保留B ep24。旧G/joint/center-only运行源码、协议和manifest未改。另一对话的可靠性源码与本文R1部分改动予以保留，不归为本轮修改。

上传以下四个运行文件，保持相对位置；不制作源码压缩包：

| 文件 | SHA256 |
|---|---|
| `crane_project/tools/check_port_train_geometry_evidence_v1.py` | `2c90b23cf482c3403c153f54144a9f11b0c16b57912cbcdfee3879e74611c826` |
| `crane_project/utils/port_train_geometry_evidence_v1.py` | `3a297fe967c915b2f8075733dd98e38f0f28de493ba2e927cd0b2f4d931f3a0c` |
| `crane_project/tools/port_train_geometry_evidence_v1_protocol.json` | `00f8a9faf0b2f44f988c853efd53fe2252119413d8bd02571a376b6d02a259d9` |
| `crane_project/tools/port_train_geometry_evidence_v1_sources.json` | `4051f9c8cb804a424d0f6fdf8bfa54bd0634c86595ab8635b069054a17e9ebd5` |

新增测试`tests/test_port_train_geometry_evidence_v1.py` SHA `a7356bbb081b81067b3f8cdb3c9c3f908839dcc0ee735a81a140dab3d591579d`；服务器运行不依赖测试文件。

新manifest复用旧62来源并加入新入口/util/protocol，共65来源，旧center-only manifest SHA仍精确`8049731476d380866e8969448c0fb54d18787340e542ff793bab34e5b72f9017`。复用原baseline SHA `ce38c0f399027de2a72bad8ff94fa5691cb1aa0d1e66048c8ef32d025487bc71`、joint SHA `7adb734eae56d703584f2a1a0d025465c07bf80b6ddb18612df0cdb07bf10452`、center-only SHA `5f7c70f1f3e65147d9d568fbbe9da23591bf7bd2d83ef68a6b59ad5bdda6b081`。继续核对原TRAIN annotation-set身份、64张选定图及标注字节、固定角色/两尺度/128行配对、原初始化和保存中心loss身份，不刷新旧SHA绕过差异。运行时只读取这三份报告及固定TRAIN文件，不要求B权重或缓存。

### 61.2 固定统计与选图规则（事实）

全部64图/128视图进入分析，保持fit/probe×real/sim×尺度共8基本组。另按序列、原B中心误差分段`<1/<2/<5/<15/≥15 px`、GT输入短边`<2/<4/≥4`个stride8单位、GT长宽比`<1.2/<2/≥2`分层，本次非空分层共92个；它们重叠且不是92个独立实验。这些分段仅作诊断，不能据结果改loss、删除样本或选择新方案。保留每视图目标/实际xy位移、原B短边归一化、符号计数、方向余弦、有符号残差及几何差异。

复用B、原G ordinary/aligned、joint完整框、保存joint中心＋B形状hybrid、center-only六种结果。逐框重算中心、周期角、尺寸与严格RIoU；保存原始宽高—角度—score关联。尺寸/方向的原始联合变化与仅换中心的hybrid明确区分，hybrid不是另训模型。输出覆盖、仅输出帧中心<15px正确、全帧中心正确三个分数显式保留分子分母；纯角统计和10px中心罚项协议角分列。中心代理loss仍是原B短边归一化的两项SmoothL1，各权重1/3。

每个fit/probe×real/sim组顺序选择一个center-only对B“中心更差且RIoU更差”的负例、一个两者均好的正例、一个原尺度B中心误差接近该组原尺度median的普通对照。负例按RIoU降幅、中心增幅、图名、尺度排序；正例按RIoU增幅、中心降幅、图名、尺度排序。已选图及其另一尺度排除后续槽位；普通对照按距median距离、图名排序。中心`1e-6 px`、RIoU`1e-5`容差仅用于好坏划分。空槽不补选，最多12个不同图，每图总显示原图及两尺度。实际固定清单如下，括号为触发尺度；两尺度始终都显示：

| 组 | 负例 | 正例 | 普通对照 |
|---|---|---|---|
| fit real | real_seq13_00179 (1.0) | real_seq13_00066 (.5) | real_seq01_00283 (1.0) |
| fit sim | sim_seq08_00202 (.5) | sim_seq08_00744 (.5) | sim_seq08_00190 (1.0) |
| probe real | real_seq13_00168 (.5) | real_seq05_00100 (.5) | real_seq05_00420 (1.0) |
| probe sim | sim_seq08_00709 (1.0) | sim_seq08_00168 (.5) | sim_seq08_00432 (1.0) |

例如real_seq05_00100的“正例”是半尺度，并不否认60.3所述原尺度退化。选图是结果引导的定位诊断，不是随机性能样本；不能用12张图估计失败比例或据此称稳定收益。该probe已经反复开发使用，B见过全部图，不是独立VAL。

### 61.3 原图证据、人工审查及证据边界（事实/待验证）

每图输出1260×1060 PNG，三列为原图、原尺度输入、半尺度输入，上排为未标记局部像素，下排为同范围框/中心叠加；另有整帧缩略图定位裁剪区域。白色原DOTA多边形、绿色固定GT矩形、橙色B、青色joint完整框、粉色center-only。joint完整框的形状代价不能混称joint中心作用，分析JSON另有hybrid供对照。

像素视图使用原生`LoadImageFromFile`解码，BGR仅交换通道为RGB显示，避免独立Pillow JPEG解码差异；调用真实`RResize(1024,1024)`及固定`PortIsotropicShrink(.5)`，保留原生整数尺寸舍入与左上坐标约定。GT边长乘`sqrt(sx*sy)`，检测器raw w/h分别乘sx/sy；仅用于重建显示与几何核对，不改变预测/标签。显示裁剪和最近邻放大使用同一等比例仿射映射处理像素及叠加，避免独立取整宽高或贴图位置造成亚像素错位；不锐化。标出实际输入尺寸及GT短边像素/stride8单位。蓝点只是stride8坐标格，不是FPN激活，也不是检测精度下限。

原DOTA多边形依实际`CraneDataset`的`poly2obb_np(le90)`转换，报告矩形拟合角点差、面积比、图像边缘距离及本机解析与固定GT差异；不自动断言标注语义或可见性正确/错误。`human_review_template.csv`全部为NOT_REVIEWED。`index.html`说明核对对象包围范围、真实可见边界、遮挡/歧义、所需纠偏是否可辨以及缩放后的边界支持。若填写，另存`human_review_completed.csv`，保留原模板和运行完成时SHA清单；后续人工文件不自动获得已验证身份或改为审查通过。

**待验证：** 人工判读是否存在对象范围/边界一致性问题；某些中心、短边和角度修正是否有充分可见依据；真实ROI/P3是否保留对应空间响应。原报告的mask=1与短边单位数仅为复用支持证据，不证明真实特征信息充分或不足。图像观察亦不能证明亚像素真值或唯一根因；不据模型误差改标签。只有前两项仍无解释时才另设计有限特征空间核对，不立即增加损失或训练新头。

### 61.4 代码复核与本地验证（事实）

最终13项针对性CPU测试通过，约1.57s：raw宽高/角度物理等价、原生解码通道/奇数尺寸缩放与GT边长约定、原B短边/两项各1/3、分组角色/尺度及归一化符号、缺输出和条件错误三个分母、唯一/两尺度/空槽/median固定选图、标注几何与语义判断区分、不替换固定GT、像素/叠加共享仿射、保存指标错误/非有限值拒绝、自身及旧来源漂移拒绝、静态/完整入口禁止CUDA初始化/torch.load/头构建、artifact字节完整、拒绝覆盖旧输出、失败保留已生成图及零更新scope。AST按Python3.8解析、JSON解析及改动空白检查通过；没有重复旧路线全量测试。

真实静态入口在系统临时目录读取现有baseline及用户tar中两份completion，通过65来源和固定TRAIN/报告身份，完成128视图复算与选图，无像素渲染。最终完整CPU入口通过，生成12图/20个SHA绑定结果文件，状态`TRAIN_GEOMETRY_EVIDENCE_COMPLETE_HUMAN_REVIEW_REQUIRED`，所有更新、检测器前向、缓存读取、GPU使用计数均0。人工检查了real probe退化、sim probe改善和sim fit普通对照的图像排版/映射，未将此渲染检查记为标注语义审查完成。本机静态/完整入口分别约1.75/1.48s仅为执行事实，不是服务器性能承诺。

六种保存结果复算RIoU及描述性GT替换增益最大跨平台绝对差`4.329843e-6<1e-5`，其他保存几何最大差`1.387779e-17`；NumPy中心loss最大差`5.495164e-9<1e-6`，原生输入GT短边单位摘要最大差`9.837983e-8<1e-5`。未修改服务器保存值或放宽性能门槛。

**本机观察，原因尚未验证：** 64张标注字节均符合固定fixture；只有sim_seq08_00514重新解析超出旧GT核对容差，中心差`.02610309px`、周期角差`.02921982°`，其余均在旧容差内。旧服务器报告的parsed GT和固定reference差为0。OpenCV构建/最小矩形拟合数值差异是可能解释，尚未确定具体原因，不能称标注错误或当前退化根因。本脚本单列`native_parse_reference_review_images`并记录OpenCV版本/build SHA，评价与显示绿色GT仍沿用原实验固定reference；不会将本机解析静默替换为新监督或据此调整方法。

本机没有服务器真实特征payload，未执行第三项或重放模型/头；服务器当前环境执行仍待用户回传。只读CPU脚本没有CUDA分配路径，命令另隐藏CUDA设备；不提供正式训练显存保证。临时解包/复算/证据图在系统临时目录完成后清理，不在项目增加服务器结果副本、review脚本或一次性报告；交接集中更新本文，不制作源码包。

### 61.5 服务器运行与结果压缩

在`/media/omnisky/personal_files/ljj/symEOOD`、原`mmrotljj`环境，上传61.1四个运行文件，保留旧源码/结果/数据。先做静态核对：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/check_port_train_geometry_evidence_v1.py \
  --check-only \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --joint-report work_dirs/port_geometry_g_center_v1_train/completion.json \
  --center-only-report work_dirs/port_geometry_g_center_only_v1_train/completion.json \
  --out-dir work_dirs/port_train_geometry_evidence_v1_static
```

预期`STATIC_TRAIN_GEOMETRY_EVIDENCE_COMPLETE_NO_PIXEL_RENDER_NO_UPDATES`。再生成有限TRAIN证据图和完整报告（仍为CPU、零更新）：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/check_port_train_geometry_evidence_v1.py \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --joint-report work_dirs/port_geometry_g_center_v1_train/completion.json \
  --center-only-report work_dirs/port_geometry_g_center_only_v1_train/completion.json \
  --out-dir work_dirs/port_train_geometry_evidence_v1_train
```

预期`TRAIN_GEOMETRY_EVIDENCE_COMPLETE_HUMAN_REVIEW_REQUIRED`，不表示发现根因或允许训练。结果包含analysis/selection、12个panels、坐标映射manifest、HTML索引、人工模板、completion/progress/artifacts；静态不含像素图。输出目录存在即拒绝覆盖，重跑需另取新目录名并同步tar参数；来源/SHA不符须解释具体差异，不刷新旧manifest来放行。原MMCV版本弃用警告本身不表示失败。

```bash
tar -czf "work_dirs/port_train_geometry_evidence_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs port_train_geometry_evidence_v1_static port_train_geometry_evidence_v1_train
```

回传该结果包即可，不含B权重、ROI缓存或源码包；人工记录可另存后随包附加。先审查全部分组及正/负/普通原图证据，再决定是否需要第三项空间特征核对。继续保持等比例变换、原图坐标、深度接口约束；未证明深度精度或连续性提升。TEST已多次暴露，本次无TEST调参/选权/选方法。

## 62. TRAIN几何监督与像素证据回传：数值核对正确，空间定位机制仍待取证（2026-10-03）

### 62.1 来源、执行范围及独立复核（事实）

用户回传`/Users/mac/Downloads/port_train_geometry_evidence_v1_review_20261003_144301.tar.gz`，5,133,986字节，SHA `62e34f7860b40473cbebaa1e91b22a0d6f46ca39202588a7535aac50370fd075`。25个普通文件＋3个目录，无重复路径/链接/越界路径；静态5个、完整20个artifact清单与所有文件SHA精确一致。静态completion SHA `e0ddb8bb83937cb191225ef39da3c0bc47e8b1ae8cde2ea50980f4172100bcf8`，完整completion SHA `9224032e9f6ef16665d590e2f30e8e85e443159b76175720ba9490e02f4388cd`。两份analysis字节完全一致，SHA `419ef9128ea9c8c4db05140e1e79adeb83e116f7bb2ca76d7a184e3a7ff78b77`。固定protocol、65来源及旧baseline/joint/center-only身份与61节一致，全部64 TRAIN图及标注字节/角色/原初始化关联通过。

静态/完整状态分别为`STATIC_TRAIN_GEOMETRY_EVIDENCE_COMPLETE_NO_PIXEL_RENDER_NO_UPDATES`、`TRAIN_GEOMETRY_EVIDENCE_COMPLETE_HUMAN_REVIEW_REQUIRED`。代码与scope一致：检测器前向/更新、头更新、ROI缓存读取、特征测量和GPU使用均0，无VAL/TEST/标签修改/权重导出。静态0图、完整12图，时间约5.78/8.16s；progress分别3/15条，后者含12个不同图的完成事件。选图、完整候选顺序、panel顺序与固定规则精确一致。没有新训练结果，几何变化均来自之前保存的G系列。

本地直接读取tar；旧两份completion和图像仅在系统临时目录使用。逐行核对128视图身份、GT、六种raw预测框、目标/实际位移、归一化、92分组及两尺度显示几何。按61节预先声明的容差复算，最大差`3.7182353e-6`来自重建joint中心hybrid的GT角替换RIoU增益，其他几何/分组统计一致。另用独立NumPy公式重算768组保存框的中心、宽高规范化/纯周期角、10px角罚项及两项/3代理loss，几何及loss差0；直接严格RIoU最大跨平台差`1.6103395e-6<1e-5`。独立核对92个组的实际区间成员、中心mean/median/RMSE/p90、覆盖三个分子分母及中心好坏计数通过。并非只凭完成状态或同一个汇总字典认定正确。

服务器保存框重算差为0；中心loss、GT输入短边单位摘要与旧报告的最大差分别`5.495164e-9`、`9.837983e-8`。未发现本次来源、配对、分母、几何变换或图框显示错位。运行源码未修改，同版本测试不重复运行；这不是再次完整项目审计。未回传真实ROI/P3或训练头tensor，不能称本地重放了真实特征和优化过程。

### 62.2 监督几何核对与原图判读（事实和边界）

服务器64图重新解析中心与固定GT差均0，周期角差最大约`1.52e-6°`（浮点转换），全部在原核对容差内；`native_parse_reference_review_images=[]`。服务器OpenCV build SHA `9c5ad2ab7a9bc194e749be8bcb660b12f7df66d821472620c1e914a38e3d9f6d`，本机build SHA `64c4e351791beeccda6b1f26ac20e853ea0c5f4820f2a2646208aa21c497c16b`。这加强了61.4本机sim_seq08_00514微小差异属于跨环境解析数值现象的解释，尚未定位具体底层原因；不将其当成训练监督错误或退化主因。评价一直使用原固定GT。

64个原多边形都在图像内，最小边缘距离102.30px，未发现这些样本因图像边界截断而产生的GT解析问题。多边形/拟合矩形面积比范围`.996481～.999997`，单图角点最大偏差的全体最大值`.109919px`；没有明显矩形转换大误差。这些数值只核对格式/几何一致性，不能证明每个框语义范围和细微边界都是真值。64图中两图GT长宽比略低于1.2（real_seq05_00051 fit、real_seq05_00420 probe），纯角判读应保留近方形轴不稳定的限定。

查看了12张返回图的原图、原尺度及半尺度未标记像素与框叠加；所有两尺度坐标元数据与native几何一致。没有观察到明显图框错位或框移到无关对象。部分real主体与周围框架/细杆投影交叠，sim所选图存在贯穿或邻接框区域的深色带状结构；其属于目标部件还是遮挡须依标注定义确认。半尺度的细边界更难辨认，但正例、负例和普通对照均存在这些情况，不能把该现象直接当成退化标签或根因。人工视觉判读无法认证小于一像素的真实中心差。

回传只有`human_review_template.csv`，没有填写后的人工表，所有字段仍NOT_REVIEWED。本轮上述观察记录在本文，不自动把模板改为全部通过。青色是joint完整框，含尺寸/角度变化；单独考察中心作用应使用analysis中的`joint_center_b_shape`。蓝点始终只是stride8坐标格，没有真实FPN激活含义。

### 62.3 分层证据进一步限定了当前问题（事实）

下表是已有center-only对B的probe结果，每组8个稀疏TRAIN视图，非新训练或独立VAL：

| probe组 | 中心mean px：B→center-only | 中心退化视图 | mean RIoU：B→center-only |
|---|---:|---:|---:|
| real原尺度 | 1.437614→1.601908 | 7/8 | .899582→.895451 |
| real半尺度 | 2.202637→2.371466 | 5/8 | .851549→.847346 |
| sim原尺度 | 1.024071→1.181230 | 7/8 | .939004→.928593 |
| sim半尺度 | 2.939849→2.738618 | 3/8 | .812432→.831290 |

全部probe中心改善10/32、退化22/32，原尺度14/16退化。原尺度B误差<1px的7视图中5个退化；B误差≥1px的9视图全部退化，因此失败也发生在确实需要更大纠偏的框上。原尺度GT短边≥4个stride8单位的9视图中7个中心退化；半尺度sim短边<2单位的4视图中3个中心改善、3个RIoU改善。小单元数可能影响定位，但这些证据不支持仅用“P3太粗/半尺度太小”解释全部退化。

real原尺度中心退化涉及seq01/05/12/13；seq06单个probe改善。各序列仅1～2个图，不能估计序列总体表现，但不能把失败只归到seq13的一个负例。

同图跨尺度反例：real_seq05_00100原尺度中心`.459485→1.014075px`、RIoU`.925177→.896646`；半尺度中心`1.329010→.094187px`、RIoU`.850282→.882424`。fit real_seq13_00179原尺度中心`4.321002→4.535862px`、RIoU`.815087→.807280`；半尺度中心`3.229646→.084324px`、RIoU`.840397→.941609`。同一原GT、同一头有相反效果，不能据更清晰原图或半尺度模糊直接确定纠偏成功条件；B起始残差及特征/采样位置同时变化。

本次三种覆盖均为128/128视图的100%，probe对应32/32；只说明已选稀疏视图的粗中心门槛仍满足，不能外推全视频输出覆盖/连续性。center-only尺寸和纯角精确B，协议角因10px中心罚项变化不等于方向精度提升，仍未解决完整几何目标。分组名`lt_2.0`是中心误差[1,2)的互斥段，`short_cells/lt_4.0`是[2,4)，不是累计<2/<4统计；这是阅读口径限定，成员复核正确。

**机制推断：** 当前有限头能在fit和某些半尺度样本中纠偏，但其从冻结特征/框描述到样本特定修正方向的映射没有稳健迁移。原尺度的普遍代价与之前向左偏置一致，但不能证明唯一原因是归一化、全局位移先验、空间信息不足或共享参数。监督/坐标核对未找到明显工程错误后，优先检查真实空间定位证据比继续加loss或扩大步数更有依据。

### 62.4 下一步建议（拟议，尚未实现）

保留B ep24，不放行G-center-only或其他同缓存增量版本正式训练；不新增VAL/TEST。接续60.4第三项，限定为一次**冻结B的真实ROI/P3空间响应核对**：

1. 固定本次12个图及两尺度，共24视图，不再选新“更有利”样本，不拟合头、不搜索系数，不用GT生成在线输出。
2. 优先读取服务器原CPU ROI缓存，核对实际9×9特征的空间变化与GT/B/已有细化框在采样坐标中的位置，复用既有mask与来源身份。mask=1或9×9插值不能替代真实定位信息。
3. 现有cache不保留完整P3；需要核对新采样位置时，对同24视图以B ep24、eval/no_grad逐视图提取P3，保持B预测/原图还原身份一致。只保存必要响应/采样几何/图像及小统计，不累计全FPN、不导出新权重；实际运行仍需先固定协议、代码审查和预算。
4. 用预先固定同幅度的±x/±y输入像素位移检查空间响应敏感性，并将输入/原图单位同时报告；比较这些变化是否对应可见目标边界还是周围强结构。GT只作离线TRAIN参照，不排序候选或声称响应强处就是正确中心。非零变化只说明表示有响应，不能自动证明可学正确方向或泛化；ROI近乎常量也不能凭一个通道幅度断言所有定位信息消失。

该取证用于决定是否值得另设计显式边界定位或采样机制，仍不是确认了新网络方案；中心、尺寸和sim方向的联合目标保持。若需要确认标注对象范围，依据已有规范与本次原图核对，另立修订版本，不按模型误差改标签。可靠性仍由另一对话推进；TEST已多次暴露，本次不用于调参/选权/选型，稀疏结果不保证深度精度或连续性。

本轮只更新本文，不修改运行代码或生成项目内解包/复算副本及review脚本；系统临时图像/旧报告在核对后清理，没有连接服务器。

## 63. 第三项实现：冻结B的真实ROI/P3空间响应核对（2026-10-03）

### 63.1 当前判断与实现身份

**事实：** 62节的监督、来源、数值及像素坐标核对通过，是可靠的排查证据；并没有得到新的几何优化成功结果。原center-only probe中心改善10/32、退化22/32，原尺度14/16退化；粗15px覆盖门槛全通过不能替代细定位精度。继续保留B ep24。没有证明标注、P3分辨率、采样或头的唯一根因。

用户授权接续第三项。新增下列四个运行文件，复用已有65项来源，扩展到68项；旧G/center-only/监督核对源码、协议及manifest不修改：

- `crane_project/tools/check_port_geometry_spatial_response_v1.py`：入口、逐视图冻结提取、缓存重放、来源/预算及结果记录。SHA `fe08b0b342550f27d94177471eeccf43861f437c58b1dd9677ee387f1f6dac6c`。
- `crane_project/utils/port_geometry_spatial_response_v1.py`：实际特征统计、固定平移、坐标及证据图。SHA `6a76b5d9a539e506329013878765ccf4271b65f046999b4216c7f4319a3cc320`。
- `crane_project/tools/port_geometry_spatial_response_v1_protocol.json`：固定协议。SHA `da32df3d0c20eb9194e856d87cb6a900fbcb24eff3ce5778de85b7338a302ec6`。
- `crane_project/tools/port_geometry_spatial_response_v1_sources.json`：68来源。SHA `ebfd79af4f263748666aebbd0d9b4280b12533cd0da5c0706a1a4423438acec2`。

另新增本地测试 `tests/test_port_geometry_spatial_response_v1.py`，SHA `130d7c8de7d02ebb1c84eeb9d8cb0c4ede52e0c842d0d39ad6e256f05436331b`。运行不依赖上传测试文件。本节为当前实现与验证记录，不回写历史实验成功状态。

### 63.2 固定协议与实际测量

**已实现，真实GPU数据待运行：** 固定62节12图，各scale1/.5，共24视图，保留原fit/probe、real/sim和正/负/普通身份。图名固定在入口与protocol，另外锁定已回传完整completion/analysis/selection SHA，selection SHA `1c6825cb087d230c4b2051844122ee5416c0c9d5d7f5e3f3551285c8d3d27dd8`。复查原64 TRAIN文件及标注、旧B/joint/center-only报告身份，不新选样本。旧报告用于既有事实，新的空间测量只涉及这24视图。

1. 静态检查来源、报告、选图及原缓存manifest，不加载ROI张量、不读B权重、不初始化CUDA。完整阶段读取一次原CPU缓存，验证全部128记录及artifact SHA；对选中24视图同时测量ordinary/aligned实际256×9×9 ROI与support。保存全通道RMS、逐通道空间去均值RMS的分位数、横纵邻点差，以及2D标量图。mask有效性与特征变化分别记录，不能混同。
2. 缓存没有完整P3，故以原B ep24、eval/no_grad、batch1对同24视图重新提取；每视图一次FPN提取、两次原生检测头调用（raw与rescale）验证坐标还原。raw及original预测与原缓存、原固定TRAIN预测同时核对；P3重放在**原缓存B采样框**处进行，隔离微小预测数值差异。ordinary/aligned ROI及mask逐元素须满足固定`atol=1e-5, rtol=1e-4`，记录最大绝对差及RMS差。身份或重放不符即停止，不放宽容差、不重建缓存来放行。
3. 在普通ROI上只平移采样中心：`(-1,0),(+1,0),(0,-1),(0,+1)`**模型输入像素**。保持B的w/h、角、1.5倍上下文、最小2个stride8单位及9×9网格定义；图像与检测输出不平移。每方向记录全通道差RMS、相对基线RMS、逐采样点差图、support变化，并报告原图位移`(dx/sx,dy/sy)`。中心差分为`(R_plus-R_minus)/(2 input_px)`，分别记录每输入像素RMS和乘对应`sx/sy`后的每原图像素RMS；它是表示的变化幅度，不是纠偏方向/误差梯度。分母≤`1e-12`时相对量记null，不制造非零归一化信号。
4. 在B采样区域、固定平移及离线参照中心附近读取原生P3网格，固定外扩1单元、最长≤64单元。保存实际原生索引/stride8坐标、全通道能量与空间去均值图，以及独立padding mask；不把9×9插值当新增分辨率。GT/B/已有center-only/joint框及中心映射到输入、原生P3及两种ROI的分数索引，GT尺寸沿用annotation几何均值、预测保留raw w/h-angle配对。GT只作离线显示/参考窗口，不进入B前向、四方向采样输入、候选排序或在线输出。
5. 原生输入像素、框叠加和离散特征图同页显示，像素用原LoadImageFromFile/RResize/等比例缩小，像素与框共享均匀显示仿射；原生P3矩形网格保留比例。全通道能量不标作目标概率/正确中心评分，不挑通道或热图最大值。四个差图在同视图共享色阶，各图打印数值范围；跨视图颜色不可直接比较。人工模板24行仍NOT_REVIEWED。

**范围/资源约束：** 检测器及头的优化更新0、无优化器/反传/训练新头/权重导出，24次特征提取、48次原生头调用，单进程单逻辑GPU；每视图结束释放完整FPN和图像tensor，不累计全图特征。旧CPU缓存约20.33MiB；新增只保留局部2D标量图、统计、PNG/JSON，不保存P3/通道tensor。逐视图报告实测峰值allocated/reserved（含采样），不能在本机缺权重时保证显存上限。完整运行要求与原报告python/torch/cuda/cuDNN/numpy/OpenCV和缓存mmcv/mmdet/mmrotate/GPU型号相同，物理卡号可换。失败保留已完成视图、cached_roi、progress/completion/artifacts；已有输出目录拒绝覆盖。

8个fit/probe×real/sim×scale组各3视图，逐项报告响应统计和原B保存几何。中心正确只统计有输出视图且<15px，另保留输出覆盖、全视图正确的分子/分母；缺失输出不填入条件分母。这里重放B与引用旧几何，不构成新的精度收益。两尺度相关、选图由历史结果引导，不能估计独立样本显著性或总体比例。

### 63.3 本地逻辑复核及验证边界

**事实：** 12项新CPU测试通过（Python3.8、torch1.8本地环境），覆盖解析线性场的半单元坐标/四方向±1px响应及中心差分、原图单位及奇数尺寸缩放、原raw w/h-angle与annotation大小约定、两种ROI坐标逆映射、零/常量/单非首通道信号、native patch索引/mask/预算、非有限/计算图/缓存替换拦截、静态禁CUDA/torch.load/提取、源漂移、失败时保留证据/真实计数、显示仿射与图像布局。另以受控CPU后端执行完整24视图驱动，验证24/48调用、detach/no_grad、逐视图落盘和冻结状态；该测试使用解析模拟特征与检测器，**不是实际B或GPU运行**。新三份Python AST及`git diff --check`通过。查看模拟证据图确认布局、离散单元、色阶、均匀像素仿射、框/中心/标题没有明显错位；没有生成项目内模拟结果。

本地用回传tar中的确切joint/center-only及完整监督证据报告、本机真实64 TRAIN文件，执行实际静态入口，状态`STATIC_SPATIAL_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`；68来源、B/原报告/固定12图身份和两尺度元数据通过。**本地缓存manifest取自已核对baseline内的原manifest字节内容**，SHA与服务器固定身份一致；这不是本机读取了服务器cache目录。张量缓存/B权重本机没有，真实cache bytes/ROI重放、GPU响应/峰值、对象边界的人工判读待服务器验证。报告/选图与模拟布局仅置系统临时目录并清理；不增加项目中的server_review副本或review_received.py，不连接服务器。

### 63.4 服务器指令与结果压缩

在原服务器checkout和`mmrotljj`环境上传63.1四个新运行文件，保留已核对旧来源、权重、缓存和报告。默认依赖：原baseline report、joint/center-only completion、`work_dirs/port_train_geometry_evidence_v1_train/{completion,analysis,selection}.json`，以及`work_dirs/port_geometry_g_v1_roi_cache/`；这些应仍在服务器。先静态核对：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/check_port_geometry_spatial_response_v1.py \
  --check-only \
  --out-dir work_dirs/port_geometry_spatial_response_v1_static
```

状态应为`STATIC_SPATIAL_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`，然后运行冻结提取（示例物理3卡在脚本内是逻辑0；这是诊断，不是完整训练）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/check_port_geometry_spatial_response_v1.py \
  --gpu 0 \
  --evidence-dir work_dirs/port_train_geometry_evidence_v1_train \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache \
  --out-dir work_dirs/port_geometry_spatial_response_v1_train
```

完整状态应为`TRAIN_SPATIAL_RESPONSE_COMPLETE_HUMAN_REVIEW_REQUIRED`。正常回传两个目录中的completion/progress/artifacts、固定protocol/selection/cache_manifest、完整阶段cached_roi、views/24 JSON、panels/24 PNG、groups、HTML索引及人工模板。出现FAILED同样回传，先定位重放/环境/来源问题；不刷新manifest或选择另一组样本。已存在输出目录需换新名字并同步压缩参数。

```bash
tar -czf "work_dirs/port_geometry_spatial_response_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs port_geometry_spatial_response_v1_static port_geometry_spatial_response_v1_train
```

只压缩诊断结果，不含B权重/ROI缓存/源码包。上传本地代码用已有方式，无需压缩提交文件包。

### 63.5 待验证判读与下一步边界

先确认工程身份/重放全部通过，再同时看原生P3、ordinary ROI的全通道空间变化、原图边界/周围结构及四方向响应，比较同图两尺度和正/负/普通样本，不单看热图亮度。P3变化明显而ROI变化受损时，才进一步提出有限采样/编码检查；两者都弱时，优先寻找更直接的边界定位证据；两者有响应也不能证明能正确读取修正方向，需判断它与可见目标边界是否对应。这些均为待验证分支，不能由一个RMS阈值或能量最大点判定根因/正确中心。

不放行同缓存FC头的权重、容量、步数搜索或正式训练；后续改进须在本次证据之后另定固定对照。维持等比例、原图还原和深度接口约束，未证明深度精度或时序改善。可靠性属于另一对话，保留其同时已有修改。TEST已多次暴露，本项无VAL/TEST访问，不据TEST重新选权/调参/选方法。

## 64. 冻结B真实空间响应回传：重放精确、表示有空间变化，正确定位读取仍待验证（2026-10-03）

### 64.1 来源、执行及工程复核（事实）

用户终端和 `/Users/mac/Downloads/port_geometry_spatial_response_v1_review_20261003_153922.tar.gz` 一致。包2,688,368字节，SHA `5f058474b97d79550434e8f98bd3ab412d3464c5a4e5e219c8b3d002b13a80f1`，64个普通文件＋4个目录，无重复/越界路径或链接。静态/完整artifacts分别绑定5/57个文件，全部SHA与实际成员一致（artifacts自身不列入清单）。静态completion SHA `a7eb985d31df63918ce8c20602141cc097a97f8cdc9397a14ce7adc64dc9e27e`，完整completion SHA `8f0796cab36f62e66fd023f2a07957f6bb4129309721c9ec2ab08ef1cb24c339`。完整cached_roi/groups/progress SHA分别 `26af8a24f0bac720f606150efe547e3e67fb1d586d705acee9f731353598c8b3`、`b06329e473c92190b51e351da5bfb3a9db781a51b48dfa7a35bddc8c24a692b9`、`454ee9944782c83e28d133c1e4da0f90ee722ee7c6bfe899ae67b55d41746d0b`。

两份identity精确相等，68来源SHA、固定protocol/selection/cache manifest与63节一致；原B、旧baseline/joint/center-only及像素核对来源链保留。两尺度依原固定12图顺序，24行JSON/PNG和progress的24个view事件精确配对，cached_roi的24行与逐视图对应字段精确一致，24张PNG均1600×1160。12个本机TRAIN图及标注SHA再核对通过，原几何72份复制（24视图×B/center-only/joint）与之前analysis逐字段精确一致。人工表24行/72个审查字段仍NOT_REVIEWED；本节人工图像观察不自动把模板改为完成。

静态状态`STATIC_SPATIAL_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`，约5.60s，GPU/缓存加载/提取/更新均0。完整状态`TRAIN_SPATIAL_RESPONSE_COMPLETE_HUMAN_REVIEW_REQUIRED`，约17.47s，一次原CPU缓存加载、24次特征提取、48次原生检测头调用、完成24视图、单逻辑GPU0（物理3）、检测器及头更新0，无VAL/TEST、正式训练、标签改动或checkpoint导出。GPU为GTX1080，torch1.13.1+cu117/CUDA11.7/cuDNN8801，环境和缓存抽取合同一致。参数/缓冲区前后摘要精确一致，并匹配原缓存B；checkpoint meta为seed0/epoch24/iter15360及配置MATCH。

**重放结果：** 24视图模型坐标及原图坐标B预测最大差均0；ordinary/aligned两种ROI及各自support对原缓存逐元素最大差均0。相对更早固定TRAIN预测的最大中心/边长差`6.8664551e-5px`、周期角差`3.5762787e-7rad`、score差`1.1026859e-6`，均在原声明容差内，不能与“本次对原ROI缓存差0”混成同一个参照。没有发现特征重放、来源、尺度还原或预算工程问题。

实测全视图最大GPU allocated含采样`343.383789MiB`、reserved`412MiB`；24次提取＋双头前向计时合计约1.963s，完整脚本时间含来源核对、加载、图像及报告。该小预算诊断没有显存异常增长，不是以后训练/新架构的显存保证。静态“No CUDA runtime”提示出现在刻意隐藏CUDA的模式；MMCV弃用和meshgrid未来参数提示均未导致本次失败。

### 64.2 数值与坐标独立复算（事实和局限）

从返回2D标量图独立复算全部ROI/P3总RMS、空间去均值RMS、相对量、96个方向差图RMS、48个中心差分图RMS及输入/原图单位换算，最大汇总差`5.5511151e-17`。用NumPy float64复算float32 support均值最大差`2.9434393e-9`，原声明数值约定下合理；support最小`.9999999404`、均值1，平移support最大变化`5.9604645e-8`，是浮点舍入量级。全部P3 patch的native有效mask为1，没有这些视图中的padding有效性代价。

逐点中心差分幅度满足正/负位移差图所约束的三角不等式；中心差分每原图像素RMS严格为每输入像素RMS乘对应sx/sy。按独立NumPy几何公式核对B/GT/细化中心的ordinary/aligned分数网格坐标，最大差`2.3645885e-7`网格单位。raw B输入坐标转回原图相对旧保存B最大差`5.9898831e-5px`（float32映射/舍入），在原恢复容差内；annotation尺寸继续使用sqrt(sx*sy)，未重配w/h-angle。原生patch为6×9～14×33等矩形，最大462个原生单元，低于固定64×64上限；未存完整P3或通道tensor。

8组成员/全部统计mean、median、p90、RMSE独立复算差0。每组3视图，三种粗覆盖为3/3；合计输出覆盖24/24、条件中心<15px正确24/24、全视图正确24/24。这里是精确重放B的稀疏视图事实，没有新增性能提升、视频覆盖保证或深度/连续性验证。

**本地验证限制：** 返回的是特征标量图、通道统计和GPU重放结果，不是256通道tensor。可复核标量统计自洽、坐标、来源及分组，不能从RMS图重新恢复通道符号、通道邻差或重新执行真实grid_sample/优化；实际tensor重放由服务器执行并在报告内记录。未重复同版本已通过的12项测试，也没有启动额外GPU抽取/训练。本机直接读tar，图像与拼图仅在系统临时目录使用并清理，项目内没有解包副本或review_received.py；不连接服务器。

### 64.3 实际空间证据（事实与解释边界）

24视图ordinary空间去均值RMS范围`.055143～.177727`，相对总RMS范围`.506841～.865877`；全通道空间变化并未趋于常量。96个固定±1输入像素平移的ROI差RMS范围`.007001～.013756`，相对基线总RMS为`3.678%～14.623%`。这些是特征幅度，绝不是定位误差、正确概率或精度收益。

下表每组6个固定诊断视图（fit/probe各3），不视为代表性总体：

| 域/尺度 | ordinary空间RMS均值 | 空间RMS/总RMS范围 | ±1输入像素差RMS/总RMS范围 |
|---|---:|---:|---:|
| real 1.0 | .091766 | .5068～.6348 | 4.730%～7.423% |
| real .5 | .066371 | .7128～.8659 | 8.515%～14.623% |
| sim 1.0 | .116533 | .5398～.6766 | 3.678%～7.299% |
| sim .5 | .077377 | .5676～.8171 | 5.565%～12.241% |

**不能据半尺度相对响应更大认定半尺度更精确。** 分母/绝对能量与输入像素对应原图尺度均不同：real本次1输入像素约`.6875～1.875`原图像素，半尺度约`1.375～3.75`；sim分别1/2原图像素。按原图单位报告的x方向中心差分RMS均值，real `.008380→.004221`、sim `.010749→.004657`；y方向real `.007813→.004394`、sim `.009466→.004011`。归一化和单位必须同时读，仍不能把这些幅度自动等同可用定位信息量。

查看全部24视图的分组拼图，并放大核对real_seq13_00168半尺度、real_seq05_00100半尺度、sim_seq08_00709原尺度和sim_seq08_00168半尺度等。框、GT/B/已有修正中心与原生网格位置没有明显错位。多个图的目标附近全通道能量低于周围，real可见周围框架/细杆，sim可见中央深色带状/框架投影；空间去均值图在部分目标区域也有局部结构。四方向差图显示响应沿横纵结构分布，并非简单“最高亮度就是目标/正确中心”。弱能量也不是缺少所有几何特征，通道聚合图不足以确定某根杆或某条可见边界的语义归属。

正/负位移RMS图往往相似：这首先是范数不保留符号及平滑场对称小位移的数学结果，不是头已失去左右方向信息的证据。已有结果同图两尺度效果相反，不能据热图形状把纠偏失败都归于某条强背景线。固定24诊断视图原尺度中心改善3/12、退化9/12，半尺度6/12改善/6/12退化，只是复制旧center-only在本次选图上的结果，不替代原全部32 probe结论。

**机制推断：** 本次不支持“ROI全常量、采样完全无响应、padding导致本批退化”这些强解释；也没有证明ROI完全保留了足够的目标特定定位信息。当前更有依据的问题是：已有有变化的空间表示及几何描述，怎样读出与目标范围相关的细定位方向/边界？语义干扰、有限头先验/监督形式、P3空间细节与跨尺度读取都仍可能参与，尚无唯一根因。

### 64.4 复用保存框补核对监督位置支持（事实，非新抽取）

为避免下一空间目标被采样区域截断，额外仅用已有保存raw B/GT和采样公式核对四角/四边中点的**几何位置**是否在分数ROI索引[0,8]内，不读取新特征、不按结果换样本。全24当前视图：ordinary完整包含GT四角13/24、角点80/96；四边中点则96/96、24/24完整。aligned两类点均96/96、24/24完整。

进一步复用上一完整analysis的128视图：ordinary角点420/512、四角完整75/128；中点493/512、四点完整116/128，缺失中点的12视图来自sim_seq08的6图两尺度（含probe seq08_00370/00617）。aligned四角和中点均512/512、128/128完整。ordinary未旋转但使用B规范化长/短边作为x/y上下文跨度，故斜长框可能超出部分采样支持；这不等于输入图像或B预测框被裁掉。

**边界：** 仅证明几何目标位置在所采矩形内，不保证9×9点恰好命中可见边界、感受野不包含周围结构或该位置具有判别力。原aligned+FC64也未满足联合精度要求（51节），不能把这个支持差异宣布为旧G失败根因，或只切aligned就认为能解决问题。它用于选择下一目标坐标的可行范围。

### 64.5 下一项建议：显式空间位置监督的有限对照（拟议，尚未实现）

保持B ep24，不放行正式训练，不继续同P3+展平FC路线的loss系数/容量/step搜索。建议另立一个**旋转ROI内四边中点定位**的小预算检查，检验空间监督/读取是否能把实际特征转成目标特定几何位置；不是已经确认新损失或根因。

- 输入复用服务器原aligned CPU ROI，原64身份×两尺度和既有48 fit/16 probe图级划分，不重新抽全P3、不用这12个选例取代fit/probe总体。选aligned仅为固定旋转局部坐标与上述位置支持，不能将后续整体收益单独归因于aligned。
- 四个目标点从原固定TRAIN OBB计算，为相对B局部坐标的四边中点，不新增人工关键点。保留9×9空间网格，以小共享卷积预测位置分布/连续位置，使监督直接落在空间位置，避免只输出展平描述的全局中心向量。连续读出避免硬argmax整数网格量化；上采样不声称提高原生P3分辨率。具体结构、loss、连续读出及固定预算在实施前另定协议，不在看结果后搜索。
- 明确旋转周期、raw宽高交换、近方形轴的匹配规则；目标点对齐到冻结B参考轴，处理物理等价OBB点排列，GT只进入离线监督。初始化/零修正须精确还原原B，几何退化/越界处理必须预先固定，不能根据GT或新结果决定在线接受修正。
- 设原B零修正参照、真实匹配ROI与固定同域/同角色/同尺度打乱ROI对照（recipient的B几何、GT、score不换，两尺度同图配对）。这是新空间目标的信息归因对照，不重跑旧FC候选排序。相同初始状态/优化预算，无step/seed/系数选择；如只学到框/场景先验，matched未必优于shuffled，不能拿fit loss下降放行。
- 第一判读是四点误差/空间分布是否与目标对应、matched相对shuffled的probe增量；再以预先固定的几何读出离线检查中心、长短边、纯周期角及严格RIoU的联合变化，并报告中心三个分母。点误差改善不保证最终OBB提升，不把未验证的点读出部署到B。真实正式训练/后续VAL必须另定，不能凭这次短fit替换B。

文献复核： [SABL（ECCV2020，官方论文）](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490392.pdf)提供逐边空间定位、分桶＋细偏移的思想，研究对象为水平框，借鉴定位表示不照搬重评分/NMS分支；[Oriented RepPoints（CVPR2022，官方论文页）](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Oriented_RepPoints_for_Aerial_Object_Detection_CVPR_2022_paper.html)提供旋转目标的点集表示和对背景离群点的空间约束思路。这里不引入其新的样本分配/质量排序。二者没有验证本项目冻结ROI、四中点或小预算方案，不能引用为项目收益证明，也未穷尽近期文献。

保留覆盖/连续性、定位/尺寸/sim方向联合目标。下一项是测试有空间变化的特征能否被更明确的位置监督有效读取，不再追加完整审计；如果matched也无probe增量，停止该表示的同缓存扩展，另讨论更直接像素/浅层空间证据。本轮只分析及更新本文，未改运行代码或启动训练。可靠性由另一对话推进，保持其已有文件。TEST已多次暴露，本次无TEST调参/选权/选型，也未证明深度或时序精度。

## 65. 旋转ROI四边中点对照：固定设计、实现与服务器指令（2026-10-03）

### 65.1 实验定位与本轮授权

用户认可第64节方向，并授权在本地实现、核对代码逻辑、提供服务器指令。新增`crane_project/utils/port_geometry_midpoint_v1.py`、`crane_project/tools/preflight_port_geometry_midpoint_v1.py`、同目录`port_geometry_midpoint_v1_protocol.json`和`port_geometry_midpoint_v1_sources.json`、`tests/test_port_geometry_midpoint_v1.py`。集中更新本文，不改原B/D/E-H/F-S/G运行源码，不连接服务器、不新增正式训练/VAL/TEST。本机没有服务器ROI payload和所需旧baseline完成报告，真实缓存及GPU检查尚未执行。

**推断：** 当前瓶颈可概括为目标特定纠偏未稳定形成或迁移，代理目标改善未稳定成为联合几何收益；根因没有唯一确定。此项检验空间读取方案能否利用实际ROI对应信息。旋转采样、空间卷积和点监督共同变化，不能把其可能收益单独归于某个因素。四点来自OBB几何，不假定是清晰可见的实体关键点；空间先验/softmax读出、损失、预算、回退及门槛不在看结果后搜索。

### 65.2 固定数学和实现合同

- 复用原aligned CPU ROI与support，原64图身份、48 fit/16 probe划分、两尺度共128视图。matched使用本人ROI；shuffled使用原seed1704、同域/同角色的固定无自身供体双射，两尺度供体保持同图，接收者B框/GT/几何元数据和score保持。无B提取或更新，不使用12个诊断选例代替总体。
- 头输入256通道ROI、1通道support、2个固定局部xy通道，结构为`Conv1×1(259→32)/ReLU→Conv3×3(32→32)/ReLU→Conv1×1(32→4,bias=False)`，17,696参数，保留9×9空间布局，无展平FC、BN/dropout、上采样或额外特征。每张图对应一个几何点；末层bias在空间softmax中不可辨识，故省略。末层weight全零。
- B规范化长/短轴的四中点作为在线初始参照。每个通道的固定B-only空间log先验`Q_j(x)=-||x-a_j||²/(2σ²)`，`σ=1/(9-1)`（一ROI网格间距）。该先验只定位原B四点，不使用GT，不是新增教师。`P_j=softmax(Q_j+logits_j)`；连续点为`a_j+E(P_j)-E(softmax(Q_j))`。这让图的空间位置与实际ROI位置对应，且校正截断网格期望的偏差，零输出能逐位保持B。无硬argmax；概率/先验/期望校准可检查，不把它当目标概率或新原生分辨率。
- 离线目标用原固定GT生成四边中点，比较4种循环排列，按相对冻结B四点的归一化平方距离选第一argmin。预测不参与匹配，处理π周期/宽高交换/方形点集等价，不人为新增方形长轴方向标签。目标匹配与所有监督仅在offline helper中，online forward不接收GT。
- 坐标使用原aligned采样合同：canonical MODEL B旋转轴、边长×1.5并下限16输入像素的上下文。依据固定图尺寸重建原RResize＋等比例半尺度元数据，先核对raw B的中心、w/h与sx/sy还原关系，再规范化轴。原图四点按xy分别乘sx/sy进入ROI，点偏移按xy分别还原原图。现有检测框raw尺寸还原、原GT annotation的sqrt(sx*sy)缩放和深度接口均不改；点坐标变换不混作annotation框尺寸缩放。任一GT点出ROI或校准后的所需期望不可表达即停止，不能删图或限幅标签。
- 只优化四点坐标的等权SmoothL1(beta=.1)，在ROI上下文归一化xy单位中平均4×2分量，不新增中心/尺寸/角度/IoU/分类损失。不保证该代理目标等于像素或RIoU，故另报原图点误差与最终几何。
- 原图矩形固定解码：`c=mean(p0,p1,p2,p3)`、`u=p0-p2`、`v=p1-p3`、`θ=atan2(u_y-v_x,u_x+v_y)`，宽/高分别为u在θ轴、v在垂直轴上的正投影。此为固定正交化读出，不声称全局最优矩形拟合。输出保留原B的raw宽高关联及最近周期角；零修正直接返回原B。另报点到正交重建点的RMS，观察四点不一致/多峰平均的代价。
- 若候选非有限、边长不正、方向退化、任何预测点出ROI，或中心移动>.30×B短边、raw边长比出[.8,1.25]、周期角改动>10°，整框回退原B。界限沿用既有有限G/joint范围，在结果前冻结；没有GT在线接受、按分量限幅或score重排。点loss不经过回退分支；报告原始候选、实际输出、所有失败原因与回退数，避免只看回退后“安全”结果。候选解码无效时单列invalid，原始候选指标不当真实输出覆盖。
- 两臂共用seed1703同一个初始state，原reviewed balanced fit-only batch8（4real+4sim）与200步schedule，Adam lr.001/wd0/clip10。仅评initial与固定final200，记录全部更新和梯度；首步四点末层梯度须有效，第二步stem梯度须有效，全程有限。不存在中间step/seed/σ/阈值/系数选择，拟合后不导出权重。

### 65.3 来源、评价、记录与继续研究条件

新manifest绑定61个来源，继承既有ROI消融/g来源链；旧baseline SHA`ce38c0f399027de2a72bad8ff94fa5691cb1aa0d1e66048c8ef32d025487bc71`、cache manifest SHA`7dae321ea2c0fb3f60eb4eb20cee9ebfb8f404d5c0780308485a24ed16f5108d`固定。cache逐文件SHA和128行B/GT必须与旧baseline一致；运行环境沿用原python/torch/CUDA/cuDNN/NumPy/OpenCV字符串，允许物理卡编号不同，无运行环境跳过开关。静态只核对来源/TRAIN/baseline/protocol，不读取ROI、torch.load或初始化CUDA；完整只读一次CPU缓存，依次拟合两臂，B根本不实例化。

所有fit/probe×real/sim×原/半尺度均报告：四点原图px及归一化ROI误差mean/RMSE/p90、位置分布和熵、原始候选和回退后中心/长短边/纯周期角/严格RIoU、正交化残差、实际/所需有符号中心移动。三个中心分母为输出覆盖、仅输出的条件中心<15px正确、全帧正确覆盖；稀疏结果不计算视频时序指标，纯角与10px协议惩罚分开。

固定继续研究门槛：matched在四个probe组的点mean/RMSE（px和ROI两单位）相对B与shuffled均不退，聚合点误差相对shuffled改善；回退后matched对B的四组中心mean/RMSE/p90、两边mean、纯角RMSE/p90、RIoU mean/p10/min均不退，无新增中心失败/RIoU<.5/零交叠输出。至少一个real组中心mean与RIoU同时改善，至少一个sim组纯角RMSE与RIoU同时改善。容差固定center/point px1e-6、point ROI1e-8、edge fraction1e-6、angle deg1e-5、RIoU1e-5，仅数值容差，不称稳定/显著收益；无论通过与否`formal_training_approved=False`，由回传人工判读再决定后续。

completion与matched/shuffled结果保存完整逐视图几何、分组、日志；`matched.spatial.json`/`shuffled.spatial.json`保存四张9×9概率图、参照/neutral/目标/预测ROI点和坐标元数据。还保存protocol/sources/cache_manifest/donor_assignment/progress/artifacts。每个完成更新立即留progress，臂完成立即独立留档；失败保留已完成证据。输出目录禁止覆盖。未返回图像包或原ROI/权重，无本地服务器结果副本或一次性review脚本。

### 65.4 本地检查与尚待服务器验证

**已验证：** Python3.8＋torch1.8.1本地环境，30项CPU检查通过，涵盖初始化逐位B、空输出、四点B先验位置、周期/宽高交换/方形匹配、奇数图尺寸与两尺度坐标、矩形解码与七类回退、四点和stem有效梯度、输入/GT梯度隔离、供体与fit-only schedule、在ROI内但期望不可表达的目标拒绝、分母、只有loss或对置乱优势不放行、静态禁GPU/缓存、完整两臂和失败留档的CPU模拟、来源篡改拒绝。入口模拟仅两个更新/臂的合成CPU数据，CUDA编排显式mock，不算真实TRAIN/GPU通过。

另仅复用fixture中64图的原保存B/GT，构造其确定性缩放元数据（没有真实ROI/P3），128视图四点均在范围内，且校准后的所需期望均可表达：最大目标坐标绝对值`.444155514`、最大所需期望绝对值`.435608983`，均<.5；零头128框精确保持这些保存B。此项是几何可行性核对，不等于真实ROI缓存重放。3个Python文件通过3.8语法检查；新61来源SHA与固定protocol检查通过。修复了本地较旧torch的多轴all/空张量min兼容性以及entropy序列化括号，最终代码以本节版本为准。

**待验证：** 服务器真实baseline/cache scope核验、真实四点目标可表达性、GPU两臂各200步、真实梯度/裁剪/显存轨迹、probe图像对应增量与最终联合几何收益。本地通过不替代服务器通过。CPU缓存仍约20.33MiB，新头参数约69.1KiB，不加载B/FPN或全TRAIN图像；结构上没有新增大模型显存负担，实际allocator峰值由服务器head_costs报告，不保证具体显存或未来正式训练开销。

### 65.5 上传文件、服务器检查与结果压缩

按项目相同相对路径上传65.1的五个新增文件；既有旧baseline/cache和来源文件保持原审查版本。无需源码压缩包。以下在`/media/omnisky/personal_files/ljj/symEOOD`、原`mmrotljj`环境执行。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_midpoint_v1.py \
  --check-only \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --out-dir work_dirs/port_geometry_midpoint_v1_static
```

应为`STATIC_MIDPOINT_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`，静态不需要ROI payload/显卡。若来源不符，核对上传版本和active checkout，不刷新manifest来放行。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/preflight_port_geometry_midpoint_v1.py \
  --gpu 0 \
  --baseline-report work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache \
  --out-dir work_dirs/port_geometry_midpoint_v1_train
```

这是一项有限对照，用一张物理卡3映射逻辑0依次执行两臂，非正式全TRAIN；不要用torchrun/DDP。应为`TRAIN_MIDPOINT_CHECK_COMPLETE_REVIEW_REQUIRED`，共400个头更新、一次缓存加载、B前向/更新0。该状态是工程完成，不能自动判实验收益。已有输出目录需另取新名字并同步下方tar参数。

```bash
tar -czf "work_dirs/port_geometry_midpoint_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs port_geometry_midpoint_v1_static port_geometry_midpoint_v1_train
```

只压缩诊断结果目录，不打包B权重/CPU ROI/源码。失败也保留并回传现有目录，先检查来源、数值与执行预算。B继续保留；probe是反复开发用的TRAIN子集，TEST已多次暴露，本轮不访问TEST/VAL、调参、改阈值或重新选权，不声明深度或连续性精度改善。

## 66. 四边中点服务器回传：工程通过，联合几何条件未通过（2026-10-03）

### 66.1 来源、执行身份与工程结果（事实）

用户回传`/Users/mac/Downloads/port_geometry_midpoint_v1_review_20261003_164837.tar.gz`，2,136,448字节，SHA `43b5e8b455192f5ba19d24e5eff7ebee6074f3a5f849087a757c219325f434ac`。直接在内存读取16个普通文件＋2目录，无重复、链接或越界成员；静态artifact绑定4文件、完整artifact绑定10文件，成员、大小及全部SHA精确一致。静态completion SHA `f3adf6447c442a49ee6d2346f8656dfab99bcf644428e94f9a2833bd7f7086de`，完整completion SHA `a3d980c8d88c978caed4c669cf409a50e5467daed3f191f7c817ba5d83414005`。

两次protocol/sources与本地当前版本完全一致，61项来源、原baseline、cache manifest、128行身份/顺序、固定48 fit/16 probe图划分、两尺度元数据、固定供体及200个平衡fit-only minibatch均核对通过。两臂相同初始state和初始化结果；各200更新事件与全部逐步日志一致，不存在中间step选择。cache manifest仍为第65节固定SHA，CPU原ROI共21,316,608字节。cached B state只是旧缓存绑定身份，此次没有实例化或重新测量B。

终端与报告状态一致：静态`STATIC_MIDPOINT_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES`，耗时2.803s；完整`TRAIN_MIDPOINT_CHECK_COMPLETE_REVIEW_REQUIRED`，耗时30.729s，缓存读取1次、头更新400、B前向/提取/更新全部0，无正式训练/权重导出/VAL/TEST。所有监督点在aligned ROI内且校准后的期望可表达，未删图或截断目标。环境python3.8.20、torch1.13.1+cu117、CUDA11.7/cuDNN8801、NumPy1.24.4、OpenCV4.13.0、GTX1080；物理卡3映射逻辑0，与固定运行身份相符。静态隐藏GPU的提示、MMCV弃用及meshgrid警告未造成该次失败。

matched/shuffled头阶段分别12.924/12.170s；两臂PyTorch峰值allocated均2.048MiB、reserved均4MiB。这是未加载检测器的有限头阶段，不能当nvidia-smi整进程占用、正式训练显存或时延。梯度全程有限，裁剪0/200；matched总梯度norm最大`.00686515`、median`.00262567`，shuffled最大`.00702081`、median`.00158922`。首步四点末层梯度均有效，第二步stem梯度matched约`4.53e-6`、shuffled约`5.10e-6`。没有数值失稳、梯度断开或裁剪压制证据；小norm本身不能推断监督无效或据此提高系数。

### 66.2 本地独立复算与正确性边界（事实）

从两臂保存的全部256个final逐视图记录及4×9×9空间分布复算：概率非负/和为1、B高斯先验及neutral期望、校准后的连续点、熵、GT四中点循环匹配、原图↔ROI xy变换、点误差、固定矩形解码、正交重建残差、七类回退与实际输出。检查全部initial/final分组、逐视图paired changes、中心三个分母及固定review门槛；分组独立NumPy统计与报告一致。SmoothL1从保存坐标独立复算，8个分组最大差`3.53e-10`。

float64本地公式对服务器float32的最大差：概率和`1.18e-7`、neutral/readout ROI坐标约`1.24e-7/1.07e-7`，GT原图点`5.82e-5px`、映射ROI坐标`1.41e-6`，候选中心/边长`9.16e-5px`、角`1.53e-7rad`、正交化RMS`3.75e-5px`。保存框的中心/边长/角度误差复算一致；严格RIoU最大跨环境差`2.41e-6`、单分量oracle增益最大`3.35e-6`，均小于原`1e-5`数值容差。没有放宽身份或实验收益门槛。

未发现本次实现的坐标还原、点排列、概率读出、候选解码、条件分母或汇总错误。全部32 probe均没有回退；matched仅有一个fit半尺度`real_seq05_00485`触发原固定angle_bound，shuffled全128均无回退。故probe代价来自实际修正，不能归于回退分支挡住改善。

**验证限制：** 回传没有真实256通道ROI payload及学成头state。本地可复算已保存的点/分布/框/汇总，实际特征张量、反向梯度与200步优化由SHA绑定的服务器代码及报告提供证据；不宣称本机独立重放了这些tensor或训练。没有重跑同版本已通过的30项CPU测试。未修改运行代码、不连接服务器，不在项目解包或新建一次性review脚本；只更新本文，并保留另一对话的可靠性记录。

### 66.3 固定probe联合几何结果（事实）

每组8视图，合计32视图来自16个TRAIN图的两尺度；这是多次用于开发的probe，不是独立VAL。表内长/短边为对应规范化边的平均相对误差%，角为**纯周期角RMSE**，不含10px中心触发的协议罚角。箭头均B→matched实际输出：

| probe组 | 中心mean px | 长边mean % | 短边mean % | 纯角RMSE ° | mean RIoU |
|---|---:|---:|---:|---:|---:|
| real 1.0 | 1.437614→1.586538 | 3.649446→4.151053 | 3.601115→4.209095 | 1.433263→2.122256 | .899582→.889469 |
| real .5 | 2.202637→2.469958 | 4.623077→4.117258 | 5.278879→5.787831 | 4.826272→5.332006 | .851549→.844815 |
| sim 1.0 | 1.024071→1.588561 | 1.869049→1.883254 | 1.005218→1.201411 | 1.198375→1.479565 | .939004→.922662 |
| sim .5 | 2.939849→2.613302 | 7.834475→6.027778 | 5.874214→5.931094 | 1.562704→1.425376 | .812433→.825087 |

sim半尺度中心、长边、方向及重叠有局部改善，但短边仍略退。其他三组点mean/RMSE两单位都相对B退化，real没有一组中心与RIoU同时改善。固定`diagnostic_continuation_conditions_met=False`、`formal_training_approved=False`复算成立。无新增probe中心≥15px、RIoU<.5或零交叠尾部失败，不能据此忽略平均与分位精度代价。

本次B/matched/shuffled各自probe输出覆盖为`32/32`，条件中心<15px为`32/32输出`，全视图中心正确覆盖为`32/32总视图`；全128视图三个指标各为128/128。分母按定义单独报告，不能把稀疏已选图100%外推全real视频覆盖/连续性，也不能用粗15px门槛代表细定位改善。

| 聚合指标 | B | matched | shuffled |
|---|---:|---:|---:|
| fit点SmoothL1 | .005247566 | .002876460 | .003471464 |
| fit中心mean px | 2.358439 | 1.724162 | 1.888144 |
| fit mean RIoU | .871079 | .900692 | .890143 |
| probe点SmoothL1 | .005618938 | .005652509 | .006286096 |
| probe点mean px | 2.586512 | 2.785255 | 3.001503 |
| probe点mean ROI单位 | .03555792 | .03683460 | .03952240 |
| probe中心mean px | 1.901043 | 2.064590 | 2.138625 |
| probe中心RMSE px | 2.397828 | 2.412903 | 2.681337 |
| probe mean RIoU | .875642 | .870508 | .862463 |

matched fit点loss约下降45.2%，全部4个fit组loss下降，中心改善71/96、退化24/96、回退保持1/96；RIoU改善75/96、退化20/96、保持1/96。说明该实现能有效拟合，不能以最后一个minibatch loss与首步比值代替固定集合评价。

probe matched点px均值约升7.68%，中心改善12/32、退化20/32；RIoU改善14/32、退化18/32。matched聚合点/中心/RIoU优于shuffled，提供本人ROI对应信息的相对增量证据；但real半尺度的点px mean/RMSE、中心mean及RIoU反而劣于shuffled，另三个组纯角RMSE也均劣于shuffled。不能称四组全面信息优势、稳定有效或显著收益。shuffled仅作归因对照，不选其上线或按域/尺度拼最优方法。

### 66.4 本次进一步限定的问题与机制（事实、推断）

**事实：** real原尺度实际中心移动均值`(-.462593,-.075844)px`，所需GT−B移动均值`(+.695728,+.245003)px`；8/8都向左修正，实际上仅3/8需要向左。real半尺度实际`(-1.051846,+.461220)px`、所需`(-.183266,+.590557)px`；7/8向左而5/8需要向左。换为保留空间布局的卷积头与等权点监督后，旧中心纠偏的左偏现象仍出现；不支持只把失败归于旧展平FC、固定中心或中心/形状loss竞争。

仅用保存的概率图补核对读出行为：matched probe 128张点分布的平均总变差距离相对固定B先验约`.111907`，16/128的最大概率bin改变；按固定四邻严格局部极大定义，0/128有多个局部峰。fit相应平均`.121095`，82/384改变bin、2/384多个局部峰。连续期望可以在峰bin不变时移动，不能说头没有学习；这些格点统计也不能认证真实边界或完整连续单峰性。当前没有广泛probe多峰平均导致退化的直接证据。

matched probe原点mean误差`2.785255px`，固定正交重建点mean约`2.766275px`；四组正交化残差RMS均值约`.491/.795/.864/.861px`。四点不完全构成正交矩形，但原点在三组已经劣于B，且中心是四点平均、在方向正交化前就已退化。仅更换矩形解码不能解释或解决这次全部代价，不据该结果搜索另一个更好解码/σ/回退阈值。

**有依据的机制概括：** 多种头与监督能拟合fit，本人ROI有部分对应增量，但从冻结局部表示到目标特定细修正方向/幅度的映射仍未在这批probe形成联合收益。点监督改善不是必然的完整OBB改善。当前B残差较小、有限fit覆盖、框/场景偏移先验、特征语义与细空间信息、归一化代理目标仍可能共同参与，尚未分离。没有证明标注错误、P3必然太粗、某个单损失或唯一几何退化根因；这次是有用的负结果，不能据fit成功直接转完整训练，也不能宣布P3完全没有可用信息。

### 66.5 下一步建议：浅层空间信息补充的有限对照（拟议，未实现）

**决策：** 保留B ep24，不正式训练当前midpoint v1，不新增VAL/TEST。按64.5的停止约束，结束当前P3缓存上的头容量、点loss、读出、σ、步数、seed及回退范围搜索；不追加同一内容的完整审计。

建议下一项只检验**现有浅层C2是否为P3提供额外的目标定位信息**。当前`crane_symeood_k1.py`的ResNet输出四个stage，而FPN `start_level=1`、检测首层stride8；C2是stride4的256通道backbone输出，未纳入当前FPN。它是已有B骨干特征，不是已训练好的P2，也不自动具有P3的语义；加入后可能同时带来细节和背景干扰。

拟议范围：冻结同B、保持同64图/128视图与划分、相同原图物理ROI范围/9×9网格、四点目标和固定200步预算。主因素为第二路特征来源，使用等容量`P3+P3`与`P3+C2`控制，避免把输入/参数增加直接归因为浅层信息；各自保留matched/shuffled信息对照。在线采样只用B，GT仍只作离线监督/评价。新C2需要有限重提取，旧结果优先复用，禁止以GT或probe误差选择层/通道；相同空间范围不强行沿用stride8的C2采样索引。联合门槛保持当前严格定义，不在看到新结果后放宽。

**实施前尚须固定：** 两路数值尺度处理、明确参数数目/初始状态公平性、供体双路同步规则、C2/P3坐标重放及padding身份、提取/CPU缓存/显存预算、来源与失败留档合同。若需特征尺度统计，仅由fit定义，不使用probe/VAL/TEST。本节是方向建议，不是已实现或已定版协议，本轮不修改运行源码或给尚未实现的服务器命令；新四臂是相同容量下的一个信息来源因素，不把旧17,696参数头成绩直接当新增容量的公平主对照。

参考[FPN原论文（CVPR2017）](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)的高分辨率层与自顶向下语义结合思想；它不证明本项目raw C2一定能改善定位，也不验证本拟议四点融合方案。[SABL（ECCV2020）](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490392.pdf)支持将边位置作为明确的定位对象，但本次负结果已说明不能把论文一般思想当成当前冻结ROI中的收益保证。文献只支持该对照问题的合理性，根因与收益仍需实证。

若这项也没有预先定义的对应信息及联合probe增量，停止这批已反复使用小样本上的增量头探索，转向另立更具代表性的TRAIN开发/独立确认设计；不以此批或已暴露TEST继续挑架构。可靠性由另一对话推进。当前TEST已多次暴露，后续也不能用于调参、重选权重或选型；维持等比例输入、raw宽高/角关联、原图还原及深度接口，稀疏本次结果不证明深度或时序精度。

## 67. 用户授权固定中点方案的探索性TEST：精确恢复头、冻结比较（2026-10-03）

### 67.1 当前效果与新增授权边界

**事实：** 第66节matched在fit及sim半尺度probe有局部收益，但real两尺度及sim原尺度多数退化，固定联合条件未通过。不能认定为有效的整体几何改进，不能仅据此放行完整TRAIN。用户随后明确“test测试也可以尝试一下”，本轮据此实现当前midpoint v1的固定TEST补充；不是提前实施尚未定版的C2方案，亦不改第66节数值、门槛或失败结论。

原入口明确`checkpoint_exported=False`，`fit_arm`返回结果和空间图后丢弃学成头；结果JSON没有足够信息恢复卷积权重。不能直接套用旧B/F-S的TEST命令冒充当前中点测试。因此仅新增下列四文件，原61项来源及其manifest/protocol保持字节不变：

- `crane_project/tools/eval_port_geometry_midpoint_v1_test.py`：原matched精确重放、头保存/重载冻结、TEST逐帧推理及统计。
- `crane_project/tools/port_geometry_midpoint_v1_test_protocol.json`：固定探索性TEST合同。
- `crane_project/tools/port_geometry_midpoint_v1_test_sources.json`：继承原61来源并绑定原manifest及三个新增源码/协议/测试路径，共65来源。
- `tests/test_port_geometry_midpoint_v1_test.py`：本地CPU回归检查，服务器运行只核对其字节、不要求运行pytest。

原matched最终state固定parameters SHA `6ac5486e6f63f1a086a068fdd901e09ff62b07ccd5ad3f6f70a666bb0f68fbfa`，buffers为空的SHA，17,696参数。原完整completion必须为第66节SHA `a3d980c8d88c978caed4c669cf409a50e5467daed3f191f7c817ba5d83414005`，不能换成另一个更好run。原checkpoint=B ep24、原TRAIN缓存与固定64图身份均继承。

### 67.2 恢复、冻结与评价的实现合同（已实现，真实运行待验证）

1. 静态只核对原TRAIN/baseline/completion、原及新增来源和协议，不加载ROI、权重，不初始化CUDA或读取TEST数据。原completion记载联合probe失败，本入口必须保持这个身份。
2. 完整阶段读取原CPU ROI一次，只原样重放matched200个fit-only更新；相同seed1703、初始state、原200个平衡minibatch、等权点loss、Adam/clip、初始及final评价。复用原batch/evaluate；恢复函数与原`fit_arm`的AST逐语句比较，唯一允许差异为最后保留head而非删除head，训练步骤本身不能变。shuffled不重跑；没有新增24epoch训练或其他拟合选择。
3. 原python/torch/CUDA/cuDNN/NumPy/OpenCV及GPU型号须相同；实际检测器的mmcv/mmdet/mmrotate等版本还与原特征缓存的extraction_runtime核对。最终**整个result（含全部200日志、128行初始/final框、分组及最终state）必须与原matched精确相等**。不符立即失败，在此之前没有TEST读取；不根据新结果放宽容差或选择种子。
4. 核验后保存单独的`fixed_matched_head.pth`，约69KiB参数加少量合同元数据，明确diagnostic_only/formal_training=False。用文件stream执行torch.save，原子发布且拒绝覆盖，重载逐tensor精确核对，恢复后的head state再次一致才冻结；清除梯度。它是原有限48 fit图×两尺度×200步头，不是完整TRAIN、经VAL选权的新检测器。
5. 之后核对固定TEST全部1440图（real_seq03=200、real_seq04=668、sim_seq09=572）的manifest/图像/标注字节。annotation set SHA `e0dbb1bd8aea7209314d8ed60bc44e965550ed606135cc0016e1075d717de13e`，image identity SHA `6448e47b9a715f47f04eacaaabb54a32b522c72208bf6e82793284de2aad665d`。不删帧、换split或创建更有利子集。
6. 加载原B ep24，state必须等于原TRAIN缓存的B state，eval/no_grad、参数全冻结。使用原单尺度TEST pipeline，每帧一次特征提取、三次原生头调用（raw/original/raw-after），比较同次B原框与matched修正框，不将旧历史TEST的微小数值差异混作改进。核对原图还原及score一致，修正后B raw逐位相同。在线API只有冻结P3/aligned ROI/support、B框和变换元数据，不传GT、域、序列或误差进行接受；GT只在预测完成后进入离线记录。
7. 缺失B输出保持缺失；有输出才执行中点头，所有原固定回退界限不变，不增加门控、候选重排或时序平滑。报告raw candidate及实际完整回退输出，不能用GT挑修正。逐帧记录fallback/invalid/理由、四点、正交残差和support，缺输出也占TEST总分母。
8. 报overall、real/sim、三序列的中心mean/RMSE/p90、长短边、纯周期角及非方形子集、严格RIoU和尾部，分别给输出覆盖、仅输出的中心<15px条件命中、全帧正确覆盖分子/分母。调用原protocol-v2完整视频统计，另报按序列及frame-ID间断切分的缺输出、中心失败、RIoU<.5连续段；原10px罚角和缺输出90°与纯角分开。v2的离线输入按等价长轴规范化，避免raw宽高交换产生虚假角度/ACI变化；记录与在线raw w/h-angle关联保持原值。无PLC/深度真值，时序指标不代表控制或深度精度。
9. TEST头更新、B更新均0，前后state及梯度核对；成功应有1440次提取、4320次原生头调用，各调用完成后即时计数/记录，失败不捏造已执行次数。逐帧释放完整FPN，不缓存全TEST特征。此次会加载完整B，不能沿用第66节纯头2.048MiB来声称TEST显存；实际allocated/reserved峰值在结果中报告。

此TEST仅补充固定短fit方案在另一已有划分上的探索性表现。无论TEST如何，`formal_training_approved=False`、`selection_on_test=False`、`original_probe_gate_passed=False`保持；不据TEST调整head、loss、回退界限、预算或选择权重，也不将一次局部改善称为稳定/显著收益、未经接触确认或替换B的依据。C2方向仍待另定设计，不能用此次TEST去挑其结构。

### 67.3 本地代码复核与限制

**已验证：** Python3.8.20、torch1.8.0.post3环境，新16项CPU检查通过，涵盖原恢复body AST相等、合成数据两步重放与原完整result/空间图精确相同、预测/头篡改拒绝、文件stream保存重载与拒绝覆盖、库版本变化拒绝、空输出/宽高交换/非完全相等sx/sy原图还原、B raw修改与未冻结head拒绝、角度界限整框回退且保留候选、中心三个分母、罚角与纯角、断帧及跨序列切段、物理等价宽高交换的v2角度/ACI一致、静态禁TEST/cache/GPU、来源失败留档以及恢复失败确实阻断导出/TEST。新两个Python文件通过3.8语法核对，原61来源字节未变，新65来源/协议核对通过；git diff空白检查通过。

新增入口SHA `357c28543e09579b06b0c0ee86b242f0ba7a0665665cb4d737e85c38e0d97361`，protocol SHA `bdc70cd3f5cf199712d225d52efcfb231085dfc72e06d183ae498c342f2695a9`，manifest SHA `b5cbc799b1aa8a6e5ff5956783e896294ef5ded3037c15a14a9e5619b1ca37c1`。原中点及G源码未修改，可靠性代码/独立交接由另一对话维护。

**待验证：** 本地没有真实TRAIN ROI payload、旧baseline报告或B ep24权重，未执行服务器真实静态入口、CUDA200步精确恢复、服务器保存重载及完整1440 TEST。合成两步/CPU测试不替代这些验证。失败目录保留已有completion/progress及artifact，不覆盖旧run或刷新原manifest放行。不连接服务器；不新增项目内服务器结果解包副本或一次性复算脚本。

### 67.4 上传、服务器指令及压缩

按相同项目相对路径上传67.1四个新增文件，保留旧61来源、原baseline/completion/cache、原B权重及全部原数据。原`mmrotljj`环境、服务器项目目录执行。先静态：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_geometry_midpoint_v1_test.py \
  --check-only \
  --train-report work_dirs/port_geometry_midpoint_v1_train/completion.json \
  --out-dir work_dirs/port_geometry_midpoint_v1_test_static
```

应为`STATIC_MIDPOINT_TEST_CONTRACT_COMPLETE_NO_TEST_CACHE_GPU_UPDATES`；没有真实TEST读取或GPU/更新。来源不符需核对实际上传/active checkout，不能刷新原或新manifest来跳过差异。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_geometry_midpoint_v1_test.py \
  --gpu 0 \
  --train-report work_dirs/port_geometry_midpoint_v1_train/completion.json \
  --cache-dir work_dirs/port_geometry_g_v1_roi_cache \
  --out-dir work_dirs/port_geometry_midpoint_v1_test_diagnosis
```

一张物理卡3映射逻辑0，无DDP。首先打印`TRAIN exact matched replay`200步；只有progress出现`TRAIN_REPLAY_EXACT_HEAD_SAVED_RELOADED_FROZEN_BEFORE_TEST`，才进入1440帧TEST。最终应为`FIXED_MIDPOINT_EXPOSED_TEST_COMPLETE_REVIEW_REQUIRED`，不是正式训练批准。完成报告包含分组/完整视频/paired变化、GPU计数和峰值；`test_rows.jsonl`留全部逐帧证据，`matched_replay.json`保留原恢复证据，单独诊断头绑定SHA/state。输出目录已经存在时另取新名字并同步tar参数，不删除旧结果。

```bash
tar -czf "work_dirs/port_geometry_midpoint_v1_test_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs port_geometry_midpoint_v1_test_static port_geometry_midpoint_v1_test_diagnosis
```

只压缩两诊断目录，包含小诊断头，不含原B权重/ROI缓存/源码包。失败也回传现有目录，先解释复现/来源/执行问题，不进入参数尝试。保留B ep24作为当前前端；TEST已多次暴露，当前工作不改变此证据边界。

## 68. 固定中点方案TEST回传：局部几何收益，联合目标仍未满足（2026-10-03）

### 68.1 回传来源、冻结身份和执行结果（事实）

用户回传`/Users/mac/Downloads/port_geometry_midpoint_v1_test_review_20261003_190010.tar.gz`与终端粘贴文件。archive为1,447,705字节，SHA `4a236f7b7bec31be9efdb2df12b2db3c617eec7f32c386c49265677933494c4e`。在内存读取13个普通文件＋2目录，无重复、链接、越界成员；静态artifact绑定4文件、完整artifact绑定7文件，成员及全部SHA一致。静态completion SHA `2404329ef6bae4d3526d3a7c50870909c8af21c6dc9c0fe66e3ac9ad3aaba4ed`；完整completion SHA `2f28c62d7ba99d68a767dec87398cc04f006bd1ea631a75168951aa7bf555f3d`。

两次protocol/sources与本地当前版本字节一致，65项来源SHA均通过，原父manifest保持不变。原fit/replay body的AST约束仍通过。将本次`matched_replay.json/result`与第66节原完整completion中`arms/matched`直接逐字段比较，整个result精确相等，包括200步日志、初始/final全部128视图及最终state；不是仅比较最后一条loss。诊断头文件102,171字节（约99.78KiB，含协议身份），SHA `9deb54785a04fe0fa347936846e7b305d6a223e0e2e3dcdd22f87cd68d5f4830`；本地CPU重载全部state，17,696参数、0 buffer，参数SHA仍为第67节固定值，checkpoint identity与本次completion一致。服务器保存/重载和冻结状态核验均通过。

静态状态`STATIC_MIDPOINT_TEST_CONTRACT_COMPLETE_NO_TEST_CACHE_GPU_UPDATES`，耗时2.794s，无TEST/cache/GPU/更新。完整状态`FIXED_MIDPOINT_EXPOSED_TEST_COMPLETE_REVIEW_REQUIRED`，耗时207.629s。progress中200个update与200个fit事件全部发生在`TRAIN_REPLAY_EXACT_HEAD_SAVED_RELOADED_FROZEN_BEFORE_TEST`之前；其后1440个提取、4320个原生头调用、1440个按固定顺序的test_frame，全部计数与报告一致。TRAIN cache加载1次、只重放原matched200步；TEST头更新/B更新0，前后state及B raw/score不变。`original_probe_gate_passed=False`、`formal_training_approved=False`、`selection_on_test=False`保留。

TEST为real_seq03=200、real_seq04=668、sim_seq09=572，共1440唯一帧；manifest/图像/标注集合身份由服务器核验并在报告记录，与第67节固定值一致。环境python3.8.20、torch1.13.1+cu117、CUDA11.7/cuDNN8801、NumPy1.24.4、OpenCV4.13.0、GTX1080，native mmcv1.7.0/mmdet2.25.1/mmrotate0.3.4，与原重放/特征缓存合同一致。原TRAIN重放耗时12.338s、纯头峰值allocated2.048MiB/reserved4MiB；TEST逐帧计时合计147.741s、峰值allocated343.451MiB/reserved412MiB。此计时包含三次原生头调用及核验，不能直接当正式部署FPS；PyTorch峰值也不代表nvidia-smi整进程占用或正式训练显存。

终端MMCV/meshgrid提示与305行list-of-ndarray建tensor警告没有导致失败；最后一项是两个scale值构造的效率提示，不能据此解释精度退化。本轮未改运行源码或来源manifest。

### 68.2 为什么有多组TEST打印，以及本地复算范围（事实）

**只做一次1440帧TEST推理。** `test_summaries`复用同一份逐帧预测，按overall、real、sim、real_seq03、real_seq04、sim_seq09六组，对B与midpoint各调用一次原完整视频评价器，所以打印12组表。overall表本身分real/sim展示，与随后单独域表重复；当前sim只有seq09，sim表与sim_seq09表完全相同。这是汇总层级重复，不是12次训练/推理、12个权重或12个独立TEST证据。第三个candidate仅保存诊断汇总，不打印完整视频表；本次所有1431个输出候选valid/accepted、回退0，因此candidate与实际midpoint逐帧一致。

本地直接读取全部1440行：核对唯一身份/序列顺序、B与midpoint缺失模式一致、分数逐位相同、B raw前后相等声明、候选/实际输出相同及全部固定回退检查通过。用保存的四点独立NumPy复算中心、对边向量、方向、宽高/原raw关联和正交重建残差；float64本地对服务器float32的最大框分量差`7.63e-5`（坐标/边长为px，角度分量为rad），残差RMS差`3.10e-5px`。从保存GT/框重新计算每帧中心、规范化长短边、纯周期角、严格RIoU及三个单分量替换oracle：中心/边/角与报告一致，RIoU最大跨环境差`5.75e-6`，oracle最大差`6.63e-6`，均在原`1e-5`数值容差内；oracle仍只是描述性替换，不能作可加的因果归因。

基于已保存逐帧指标重新生成全部六组、三种输出的完整汇总、paired变化、v2指标、缺输出/中心/RIoU失败区间，逐字段相等。另从GT/预测框直接独立复算中心均值、长短边均值及周期角RMSE，与保存分组一致。未发现本次坐标解码、原图指标、三个分母或分组/失败段的错误。

**验证限制：** 本地没有B权重、完整TEST特征/图像tensor或原真实TRAIN ROI payload，没有独立重跑检测/200步CUDA优化。服务器执行由SHA绑定代码、原结果精确重放、state和progress提供证据；本地重载了小诊断头并复算保存点/框/指标，不能声称本机验证了原生检测器的全部数值过程。该版本16项CPU检查此前已通过，本轮没有无变化地重跑。没有连接服务器、在项目解包回传文件、保存一次性复算脚本或源码包；只更新本文，保留另一对话可靠性改动。

### 68.3 覆盖与单帧几何（事实）

B与midpoint的输出覆盖、条件中心<15px命中、全帧中心正确覆盖完全相同，但分母必须分别列出：

| 组 | 输出覆盖 | 条件中心命中，仅输出 | 全帧中心正确覆盖 |
|---|---:|---:|---:|
| real | 859/868＝98.9631% | 858/859＝99.8836% | 858/868＝98.8479% |
| sim | 572/572＝100% | 572/572＝100% | 572/572＝100% |
| real_seq03 | 193/200＝96.5% | 192/193＝99.4819% | 192/200＝96% |
| real_seq04 | 666/668＝99.7006% | 666/666＝100% | 666/668＝99.7006% |
| overall | 1431/1440＝99.375% | 1430/1431＝99.9301% | 1430/1440＝99.3056% |

打印的v2 `real/R_center=98.85%`是全帧正确覆盖，**不是**仅输出的条件命中率。输出缺失保持不变是冻结B分支的结构约束；<15px的粗正确覆盖不等于细定位、尺寸、方向或深度精度被保证。

下表箭头为同次B→固定midpoint实际输出。中心/边/纯角只统计输出帧；长短边为平均相对误差%，纯角为等价长轴的周期角RMSE，排除缺输出/中心≥10px的90°协议罚角；mean RIoU为全帧、缺输出计0。

| 组 | 中心mean px | 长边mean % | 短边mean % | 纯角RMSE ° | 全帧mean RIoU |
|---|---:|---:|---:|---:|---:|
| real | 2.662757→2.421726 | 5.191685→5.253642 | 6.541198→6.504452 | 2.621029→2.809047 | .816882→.821851 |
| sim | 1.145166→1.258354 | 4.033833→3.713336 | 4.478021→3.720255 | 1.882177→1.788400 | .881540→.883693 |
| real_seq03 | 3.058756→3.217331 | 8.414425→7.417284 | 6.512309→5.694212 | 3.273134→3.044623 | .761123→.767904 |
| real_seq04 | 2.548001→2.191168 | 4.257768→4.626641 | 6.549570→6.739251 | 2.399156→2.736994 | .833576→.838003 |

real聚合中心mean约改善9.05%，中心RMSE3.176692→3.000378、p90 4.563156→4.233321；但长边均值及两边p90退化，纯角RMSE约增加7.17%，角p90 3.941210→4.492305。中心改善主要来自real_seq04，real_seq03中心mean/RMSE/p90都退化。sim尺寸均值/角RMSE与RIoU有收益，中心mean约退9.88%，RMSE1.364497→1.483421、p90 2.209309→2.452108；其长边p90 .073328→.076214、角p90 2.948730→2.972390也略退，不能说所有几何或尾部均改善。

全部输出的paired细定位计数：real中心改善546、退化313；sim改善215、退化357。严格RIoU按原`1e-5`数值容差区分：real改善506、退化353；sim改善294、退化278。没有新增/恢复中心<15px事件；real有1帧RIoU<.5恢复、1帧新增失败，sim无此事件。聚合overall mean RIoU .842565→.846416，但纯角RMSE2.353687→2.452572；不能用整体均值或进步帧占比代替逐域联合目标。

### 68.4 时序与尾部代价（事实）

real DFR 2.5106→2.6925、ACI .9440→.9401、MCML_mean 3.5→4、MRF 2.29→2.67，均有代价；TDR_w10仍100%、MCML_max聚合仍4，不能抵消其他时序退化。sim DFR 2.4734→2.4707（很小变化）、ACI .9479→.9527、MCML仍0，是局部时序收益，不代表所有场景稳定改善。

real_seq04 `real_seq04_00204`新增RIoU<.5（delta `-.0136308564`），原205～207的3帧失败段变为204～207的4帧；该序列DFR2.2013→2.4701、ACI .9526→.9479、MRF2.5→3，且最小RIoU .427034→.408561。real_seq03恢复1帧、DFR改善，但ACI/MRF略退。real聚合最长RIoU失败仍4，因为seq03原本已有4帧段，掩盖了seq04 3→4的退化；必须同时查看序列级结果。缺输出/中心失败最长段real聚合仍4、seq04仍1，三种失败的含义不同。

sim所有输出中心<10px，无协议罚角，其A-RMSE与纯角RMSE一致；real本次中心≥10px罚角计数3→4（仍都不是新增≥15px中心失败），纯角与罚角继续分开。上述是冻结方案的探索性视频指标，不外推深度、真实运动或控制安全精度。

### 68.5 综合判断与下一步边界（事实、推断、拟议）

**判断：** 本次是正确执行的固定短fit诊断，存在局部收益，不是无作用；但未形成定位、尺寸、方向及real连续性的联合改善。原反复开发的TRAIN probe条件仍为失败，TEST不能撤销它，当前midpoint v1不放行正式训练、不替换B ep24、不按域/序列拼最优框或改变回退界限。此TEST已多次暴露，又增加了一次结果暴露，不能作为未接触确认、调参、重选权重或架构选择依据。帧间高度相关，单次微小变化不称稳定或显著收益。

**机制推断：** 显式四点头可以产生有效的几何修正，但修正方向/幅度及各分量收益依场景而变；中心、形状与重叠收益并不同步。结合既有fit/probe结果，问题仍是冻结局部表示到细几何残差的可泛化映射没有形成可靠联合增量。数据代表性、局部特征语义/分辨率和代理点目标均可能参与。当前结果没有证明P3完全无信息、stride8是唯一根因、标注错误或某个损失导致全部退化。TEST中的域/序列差异仅作当前方案描述，不能据此选择域专用模块。

**下一步仍沿用TEST前第66.5节的有限TRAIN信息来源对照，未实现：** 冻结B；等容量`P3+P3`与`P3+C2`，各matched/shuffled；保持原64图/128视图、48 fit/16 probe、四点目标、物理ROI范围/9×9网格、固定200步/臂和原联合门槛。只检验已有stride4 C2能否补充P3的目标定位信息，不宣称它必能修复根因。实施前先固定fit-only数值尺度处理、参数/初始化公平性、两路供体同步、stride4/8坐标及padding核对和资源预算；禁止根据本次TEST决定层/通道/结构/阈值，停止当前P3缓存上的同头/loss/seed/σ/预算搜索。该下一项未运行，本轮不修改运行代码或提供虚构的启动命令。

这批16图probe已多次用于开发，下一项也只能作机制筛查；即使通过，也不能自动宣称泛化或转完整训练，后续须另立具有代表性的TRAIN开发/独立确认及固定VAL方案。若对应信息与联合增量仍未达到原条件，停止同一小样本增量头路线，重新处理代表性与确认设计。继续保持等比例输入、raw宽高—角度关联、原图还原及深度接口；可靠性另一个对话推进。

### 68.6 用户要求的综合TEST评价补充：整体单帧收益应被保留（2026-10-03）

用户指出本次改善项目较多，要求完整TEST指标与综合比较。本节复用同一已核验archive，不增加推理、训练、TEST选择或新权重。第68.5节的预先固定联合门槛结论保持，同时补充总体收益；“联合条件未通过”不能概括为“没有改进价值”。

**事实：** 以下overall按原逐帧集合汇总，中心/边/纯角使用1431个输出，RIoU均值使用1440全帧；不是real/sim两个均值简单平均。相对变化为本次与同次B相比的误差下降幅度，RIoU只给绝对差值，不能混成一个临时加权总分。

| overall指标 | B | midpoint | 变化 |
|---|---:|---:|---:|
| 中心mean px | 2.056145 | 1.956702 | 误差下降4.84% |
| 中心RMSE px | 2.608038 | 2.506687 | 误差下降3.89% |
| 中心p90 px | 3.824154 | 3.594103 | 误差下降6.02% |
| 长边mean相对误差 % | 4.728868 | 4.637950 | 误差下降1.92% |
| 长边RMSE相对误差 % | 6.402177 | 6.271384 | 改善 |
| 短边mean相对误差 % | 5.716504 | 5.391551 | 误差下降5.68% |
| 短边RMSE相对误差 % | 6.917883 | 6.673082 | 改善 |
| 纯角RMSE ° | 2.353687 | 2.452572 | 误差增加4.20% |
| 全帧mean RIoU | .842565 | .846416 | +.003851 |
| 输出RIoU p10 | .766910 | .772416 | +.005506 |
| 输出最小RIoU | .377537 | .401213 | +.023677 |

三条序列的全帧mean RIoU均提高，real_seq03 +.006781、real_seq04 +.004427、sim_seq09 +.002153；不是只有一个序列改善了重叠。中心/边/角不一定与每帧重叠同步；800/1431（55.90%）个输出RIoU改善，631退化（按原1e-5数值容差），761/1431（53.18%）中心改善、670退化（描述性1e-6px容差）。这些是同一组相关指标/帧的描述，不能按改善指标数量宣布显著或独立收益。

关键收益幅度：real中心mean下降9.05%、RMSE下降5.55%；sim长边mean误差下降7.95%、短边下降16.92%、纯角RMSE下降4.98%。两域RIoU p10与域内最小值均提高。全部输出及<15px全帧正确中心覆盖保持，全部三个序列TDR_w10仍100%、MCML_max仍≤5。因此本次没有整体输出/粗覆盖崩溃，存在值得保留的几何收益证据。

代价不能省略：real纯角RMSE增加7.17%、DFR相对增加约7.25%、ACI绝对下降.0039；real_seq04 RIoU失败段3→4虽仍通过原≤5粗门槛，连续性仍退化。sim中心mean增加9.88%，但绝对增加约.113px，仍未新增<15px失败；overall角RMSE增加约.099°。real长边mean略退，虽然长边RMSE下降；overall两边p90与纯角p90退化。综合解读应同时呈现绝对幅度、平均/尾部、域和序列分布，而不是只取最好/最坏一个指标。

**综合判断：** 当前固定TEST上的midpoint v1可表述为“具有总体单帧几何收益、仍有分量与real时序代价的候选改进”，可以保留其固定诊断头及结果作为研究证据。它尚未证明满足用户全部目标，亦未通过原反复使用TRAIN probe联合条件。既有B ep24保持为正式保留模型，原C2有限TRAIN信息来源对照仍沿用TEST前设计；不因为本次综合收益临时降低门槛或在TEST选型/调参。每个数字均改进不是评价研究价值的唯一方式，关键约束与预先定义的实验决策仍须单独遵守。

完整原protocol-v2输出为real的R_center/mean_RIoU/DFR/ACI/TDR_w10/MCML_max/MCML_mean/MCML_pass/MRF，sim相应输出另有A-RMSE、没有MRF（无失败恢复事件）；real不输出协议A-RMSE。DEP两域均未输出，因为没有PLC真值，不是深度误差为0。补充报告中的real含90°罚角RMSE为10.886699→11.354352，必须与纯角2.621029→2.809047分开。完整几何描述的mean/median/RMSE/p90、罚角及oracle仍在原completion中，不能用未输出项充0。本轮仅补充本文，不改运行代码、不重复已有测试。

## 69. 用户授权的独立方法快照与创新性核对（2026-10-03）

**用户最新意图：** 认可当前中点候选的综合收益，希望后续微调恢复real DFR/ACI；本轮先额外保存具体方法与结果，并判断能否作为小论文方法。没有据此修改运行源码、调整系数/阈值或启动正式训练；第66.5节C2仍为未实施建议，本轮没有自动选定下一项代码路线。

已新增[旋转ROI四边中点几何修正_midpoint_v1_方法与结果_20261003.md](detection/旋转ROI四边中点几何修正_midpoint_v1_方法与结果_20261003.md)。这是用户明确要求的独立文本快照，区别于不保存的服务器回传JSON/权重/一次性review脚本。表格从第66/68节原archive直接生成，保留方法公式、17,696参数、匹配/坐标/解码/回退、有限fit/probe身份、三个分母、完整protocol-v2、mean/median/RMSE/P90及序列结果。后续运行状态继续集中维护本文，旧v1结果不被后续微调覆盖。

**论文检索事实：** [MidNet作者原文](https://arxiv.org/pdf/2111.10961)已有中心＋四边中点、热图定位/向心匹配和解析几何构框，是直接相关表示先例。[Oriented R-CNN](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)已有外接水平框中点到旋转顶点的偏移编码，和本项目四OBB边中点不是同一表示。[DSNT](https://arxiv.org/abs/1801.07372)与[Integral Regression](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.html)已有可微空间到坐标读出，[Removing the Bias of Integral Pose Regression](https://openaccess.thecvf.com/content/ICCV2021/html/Gu_Removing_the_Bias_of_Integral_Pose_Regression_ICCV_2021_paper.html)已有softmax＋期望偏差及补偿研究。仅搜索这些直接相关原始来源，不是穷尽原创性认证，不把引用工作性能外推本项目。

**创新性判断（推断）：** 可以写入小论文的方法改进与消融部分，具有面向冻结高覆盖检测器的局部纠偏和B先验初始化中性化等具体适配差异。四中点、旋转ROI、积分坐标读出及一般偏差补偿本身不是本项目首创；减去E(P0)只证明指定B先验零修正中性，不宣称通用bias消除。原理性新意有限，组合及应用贡献仍需要最接近简化基线、关键校准消融和独立/固定VAL确认支撑；当前不足以保证整篇EAAI等期刊创新性或录用。恢复DFR/ACI也不会自动带来新颖性，不靠堆模块命名。

**后续约束：** 只在TRAIN/固定VAL设计并选择恢复real时序的方案；检查修正噪声与真实尺寸/角度变化，避免压掉真实运动或用迟滞换表面DFR/ACI。已多次暴露TEST不能用于系数/窗口/权重/架构选择。稀疏64图不当完整时序数据；具体邻帧监督/因果残差平滑尚未设计定版或实施。B ep24仍保留，原probe失败和本次探索性TEST身份均不变。


## 70. 用户授权的全TRAIN中点头正式候选：midpoint-formal v1（2026-10-03）

### 70.1 身份与本轮事实

用户要求先完成正式训练，并询问是否改变已有性能、是否包含SymEOOD及是否从B出发。本轮授权落实为**两阶段训练、完整串联推理**：第一阶段已完成的SymEOOD＋等比例尺度增强B采用VAL epoch24；第二阶段冻结该检测器，在完整TRAIN上适配一个新的midpoint头。没有服务器连接、GPU训练或新的TEST读取。本地已实现、静态核对并通过合成CPU检查；真实cache/smoke/训练/VAL仍待服务器运行。

- 检测器：B原backbone/FPN/主头/辅助分支全部eval、requires_grad=False、no_grad；B更新0，不重新计算或优化SymNFL/SymKLD/SymPOLA。
- 小头：原17,696参数旋转ROI四中点结构、B-only先验、残差连续读出、四点SmoothL1、原图还原、原raw宽高—角度关联及整框回退规则保持。
- 初始化：seed1703的新头，输出层为0，初始交付框逐位等于B。不加载原200步诊断头或其TEST权重；smoke权重也不用于正式初始化。
- 推理：图像→B及其P3→同一检测器的aligned ROI→正式midpoint头→修正OBB。仅一个检测器，没有独立模型投票/融合、GT在线输入或可靠性决策；B score/输出数量/缺失状态保持。中心坐标本身可被修正，中心正确覆盖不是结构保证。
- 正式头参数变化后，性能可能改善或退化。第68节短fit的TEST数字只属于原诊断头，不能移作正式头成绩；原probe失败仍失败，新授权不追溯修改门槛或实验结论。
- 此阶段测试“原中点结构在更完整TRAIN支持下的表现”，不是已证明数据量为唯一根因，也不夹带DFR/ACI时序补救、C2或更大网络。

### 70.2 固定训练与VAL协议

| 项目 | 固定内容 |
|---|---|
| TRAIN | 全2558图：real1810、sim748；每图1.0/.5确定性等比例视图，共5116视图 |
| 监督资格 | 沿用有限v1：B有输出且原图中心误差<15px；无输出/错位也保留在缓存并单独报数，仅不进TRAIN梯度；不筛选VAL |
| 数据增强 | 确定性两尺度、无新翻转/光照；这是头适配协议，不宣称复现B的随机0.5..1.0增强 |
| 优化 | 24轮；Adam lr.001/wd0；batch8（real4＋sim4）；clip10；同一预置seed，无系数/seed/步数搜索 |
| 每轮采样 | 对两域全部合格视图分别打乱，较小域循环补齐；每个合格视图每轮至少访问一次，无VAL优化、GT误差加权或困难TEST采样 |
| 更新预算 | 每轮ceil(max(合格real视图数,合格sim视图数)/4)步；全部合格时905步/轮、21720步总计；实际资格及步数写入报告 |
| VAL | 完整887图：real_seq07=226、real_seq14=149、sim_seq10=512；仅原验证尺度；每轮全部评价 |
| 选权 | 所有24个epoch，沿用ckpt_sweep.SELECTION_CONFIG和select_best_checkpoint；不平均权重，不用TEST挑选 |
| 保留判断 | 按选中权重另报与同运行B的覆盖/中心/尺寸/角度/RIoU/时序及逐序列变化；选出头≠自动替换B |

原规则仍为中心近最优与MCML≤5约束、原跨域权重及TDR/中心/ACI/角度软评分；原fallback也保留，不因几何结果改变选权规则。新报告另列预置非退化审查项：两域输出/全帧正确中心、无新增中心错误、中心连续性、meanRIoU、两边mean、DFR/ACI/MCML；real中心RMSE、sim纯角RMSE及各VAL序列RIoU最长失败。数值容差只用于浮点比较，不表示显著性。`all_checks_passed`是审查结果，`automatic_promotion=False`始终保留。

中心命中仅以输出帧为分母，同时报输出覆盖和全帧正确覆盖。纯周期角与中心>=10px/缺失的90°罚角分开。时序使用旧评价器`mode='test'`以请求全视频指标，实际输入、selection及报告split全部是**VAL**；不读取TEST数据，终端不重复打印各层级大表。等比例坐标、深度接口保持；未提供PLC/深度真值，不认证深度精度。

### 70.3 实现与资源约束

新增文件：

- [正式训练入口](../crane_project/tools/train_port_geometry_midpoint_formal_v1.py)：check/cache/smoke/train四阶段，完整VAL与原规则选权；`load_selected_pipeline`加载B＋VAL选中的正式小头，之后沿用GT-free capture推理。
- [固定协议](../crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json)。协议中的继承v1 settings保留历史200步字段；正式循环只读取本文件的24轮SETTINGS，不调用原200步fit/replay。
- [69项源码清单](../crane_project/tools/port_geometry_midpoint_formal_v1_sources.json)：绑定原运行文件、新入口/协议/测试及原选权源码。源SHA不符即停止，不自动刷新。
- [12项新CPU检查](../tests/test_port_geometry_midpoint_formal_v1.py)：完整合格视图覆盖、TRAIN/VAL隔离、detached缓存、GT-free原生坐标/缺失、文件流无覆盖IO、优化器/RNG保存续步、三个中心分母、两轮整合与VAL选权、选中头身份、静态零data/cache/GPU。

缓存是**新全TRAIN/VAL缓存**，不能拿旧64图缓存冒充。仅每张视图一次B提取和三次原生头核对（raw/original/raw-after），共6003提取、18009头调用、B更新0。CPU保存aligned ROI/support及必要框/尺度/GT；不保留全FPN或梯度图。局部张量总上限1GiB，按所有帧有输出估算约477MiB，另有框、元数据、模型和加载开销；不是进程RAM上限。

单张逻辑GPU负责逐图冻结提取，之后只训练小头；正式TRAIN覆盖与GPU数量没有等价关系，不用torchrun/DDP。报告实际GPU max allocated/reserved；冻结提取通常接近已有B推理资源，但没有真实服务器测量前不保证确切显存或无OOM。cache和训练要求相同Torch/CUDA/cuDNN/mmcv/mmdet/mmrotate/OpenCV/GPU身份。

强制server smoke：两步TRAIN更新→文件流保存/重载→head/optimizer/RNG逐项一致→6帧VAL实际B＋头与缓存一致、B state不变→两条同状态续步精确一致。只有`FORMAL_MIDPOINT_SMOKE_SAVE_RELOAD_PASS_DISCARDED`且实际在线6帧通过，才允许正式训练。formal重置seed、创建新头/optimizer，smoke状态不被继承。24个epoch全部保留，选中的头从文件重载后必须复现该epoch完整VAL；独立保存selection/所选VAL对照，不触发TEST。

### 70.4 本地复核事实与未验证事项

新增12项CPU检查通过；另16项既有中点TEST入口兼容检查通过，原短fit/runtime源未修改。合成两轮训练覆盖完整epoch、保存全部checkpoint/逐帧VAL、原规则选择并重载复现。首次整合测试用2帧片段没有TDR_w10，因而原选权拒绝；已将合成片段改为12帧，未改真实选权规则或给缺失TDR补0。新资源/元数据检查追加后重跑受影响12项并通过。

本地静态check不读数据/权重/缓存，不初始化CUDA；真实TRAIN/VAL字节核验、6003次CUDA提取、在线smoke、24轮训练和正式VAL结果都仍待服务器回传。不能将CPU成功写成GPU或正式收益。当前TEST多次暴露；本轮没有增加TEST访问，不保证后续一次冻结TEST是独立确认。

### 70.5 服务器运行顺序

先上传新增的入口、protocol、sources、测试四个文件，保留它们的项目相对路径。原69项依赖仍沿用服务器已经审核过的项目代码，不改B配置或原v1协议。进入当前服务器项目根目录并激活既有mmrotljj环境：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python crane_project/tools/train_port_geometry_midpoint_formal_v1.py   --stage check   --out-dir work_dirs/port_geometry_midpoint_formal_v1_static
```

通过后，建立全量CPU局部缓存。物理卡3映射逻辑0；可以统一换为另一张空闲物理卡，但cache/smoke/train需同型号及运行环境。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python crane_project/tools/train_port_geometry_midpoint_formal_v1.py   --stage cache --gpu 0   --cache-dir work_dirs/port_geometry_midpoint_formal_v1_roi_cache   --out-dir work_dirs/port_geometry_midpoint_formal_v1_cache_check

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python crane_project/tools/train_port_geometry_midpoint_formal_v1.py   --stage smoke --gpu 0   --cache-dir work_dirs/port_geometry_midpoint_formal_v1_roi_cache   --out-dir work_dirs/port_geometry_midpoint_formal_v1_smoke
```

仅当最后status为`FORMAL_MIDPOINT_SMOKE_SAVE_RELOAD_PASS_DISCARDED`时运行正式训练；脚本也会验证原报告、保存文件SHA及6帧实际在线门槛。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python crane_project/tools/train_port_geometry_midpoint_formal_v1.py   --stage train --gpu 0   --cache-dir work_dirs/port_geometry_midpoint_formal_v1_roi_cache   --smoke-report work_dirs/port_geometry_midpoint_formal_v1_smoke/completion.json   --out-dir work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1
```

最终status应为`FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED`。重点回传`completion.json`、`selection.json`、`selected_val_compare.json`和逐轮日志/VAL。输出/缓存目录必须新建；已有目录用新名称并同步后续参数，不删除或覆盖旧实验。该入口没有断点续训参数，意外中断保留已写产物，不通过改epochs/step进行临时选型。

压缩**结果文本**的指令（不打包源码、B/头权重或大ROI缓存；原文件均留服务器）：

```bash
tar --exclude='*.pth' --exclude='*.pt'   -czf "work_dirs/port_geometry_midpoint_formal_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz"   -C work_dirs   port_geometry_midpoint_formal_v1_static   port_geometry_midpoint_formal_v1_cache_check   port_geometry_midpoint_formal_v1_roi_cache/cache_manifest.json   port_geometry_midpoint_formal_v1_smoke   crane_symeood_k1_port_day2night_midpoint_formal_v1
```

训练结束并分析VAL后，再单独准备正式头的冻结TEST入口。旧`eval_port_geometry_midpoint_v1_test.py`会精确重放旧200步头，**不能拿它测试正式头**；普通`tools/test.py`也不能把head-only权重当SymEOOD整模型权重。完整部署路线有B和头两个参数文件，属于同一串联模型流程，不要求融合独立检测模型。可靠性仍由另一对话维护；采用新前端后需要针对新框另做标签/校准/评价，不能把B原框可靠性结果直接迁移为新系统收益。


## 71. 截图所示服务器work_dirs的几何结果清理范围（2026-10-03）

**用户授权与实际操作：** 用户要求给出删除无用几何文件夹的服务器指令，可靠性文件可能混在其中。本轮只核对截图完整名称、当前formal入口和旧诊断依赖，不连接服务器、不删除真实实验目录，也不改可靠性记录或运行代码。指令只处理下列12个已结束诊断的展开结果目录；先完整tar归档并校验gzip/tar可读，再以精确白名单删除展开副本，不用port前缀通配。

| 已结束诊断 | 归档后可删除的展开目录 |
|---|---|
| ROI对应信息 | port_geometry_g_roi_ablation_v1_static、port_geometry_g_roi_ablation_v1_train |
| FC16容量 | port_geometry_g_capacity_v1_static、port_geometry_g_capacity_v1_train |
| 联合中心头 | port_geometry_g_center_v1_static、port_geometry_g_center_v1_train |
| 独立中心头 | port_geometry_g_center_only_v1_static、port_geometry_g_center_only_v1_train |
| TRAIN监督/像素证据 | port_train_geometry_evidence_v1_static、port_train_geometry_evidence_v1_train |
| 冻结空间响应 | port_geometry_spatial_response_v1_static、port_geometry_spatial_response_v1_train |

**依赖核对事实：** 这12个目录不作为第70节formal入口的运行输入；旧center-only/监督证据/空间核对入口仍引用其中报告，所以重放旧诊断需先恢复归档。不能把已结束的负结果称为无研究价值。当前Downloads中容量、joint、center-only、监督证据及空间响应5份原压缩包仍存在且分别包含对应两目录；ROI信息消融原包已不在该路径，故不依赖本机下载作为删除前备份保证。

**保留范围：** 所有port_reliability_*、port_simple_reliability_*及port_direction_consistency_*属于另一个工作线；当前四个port_geometry_midpoint_v1_{static,train,test_static,test_diagnosis}及port_geometry_g_v1_roi_cache保留，原G baseline JSON和port_train_val_geometry_v1_cache也保留。所有crane_*训练目录（含B、EOOD公平对照、D/E-H/F-S/C）本轮不清理。新formal缓存/check/smoke/训练输出、tools/configs/tests及数据集全部不在删除名单。

服务器指令将备份放在work_dirs/_geometry_archive/closed_geometry_时间戳.tar.gz；任一步归档/读取校验失败立即停止，不执行删除；仅处理存在的白名单真目录，拒绝同名符号链接。没有连接服务器，未知其当前磁盘占用、内容新变化或运行任务；实际执行前不要同时启动这些旧诊断。当前formal和可靠性工作无需停止。

**指令验证事实：** 在临时合成目录执行相同归档/删除逻辑：归档中原内容可读、仅白名单展开副本被移除、模拟B/midpoint/ROI缓存/可靠性保留目录及其内容不变；同名符号链接在归档/删除之前被拒绝。临时测试自动清理，没有服务器/真实work_dirs删除。无需重复模型测试。是否已执行服务器清理仍待用户回传；不得把这份计划写成已经清理完成。


## 72. 正式midpoint的VAL复核与固定TEST入口（2026-10-03）

**用户授权与本轮事实：** 用户要求后续VAL/TEST指令。本轮新增独立评价入口，不修改第70节训练源码、训练协议/69项来源或原短fit v1。没有连接服务器、启动训练、读取实际VAL/TEST图片或生成新性能。正式训练已自动完成24轮VAL及选中头重载；额外VAL属于可选的真实在线复核，不是第二轮选权。

新增四文件：[评价入口](../crane_project/tools/eval_port_geometry_midpoint_formal_v1.py)、[评价协议](../crane_project/tools/port_geometry_midpoint_formal_v1_eval_protocol.json)、[73项评价来源](../crane_project/tools/port_geometry_midpoint_formal_v1_eval_sources.json)、[CPU检查](../tests/test_port_geometry_midpoint_formal_v1_eval.py)。需要按项目相对路径上传；原正式训练来源保持，不刷新旧manifest放行。

**入口逻辑：**

1. 检查完整24epoch正式completion、selection、artifact及选中头SHA；从保存的24组VAL指标重算原选权，必须仍为相同头、相同selection_info。不能传任意epoch或短fit头。
2. `--check-only`只读取源码/protocol、训练结果JSON和选中权重字节SHA；不torch.load、不加载缓存/图片/标注或初始化CUDA。此阶段不代表已完成真实数据/GPU验证。
3. 正常运行加载冻结B ep24＋选中正式头，核对原head/B state、cache身份及原运行库/GPU型号。B/head更新0，禁止DDP，不做任何TRAIN重放。
4. VAL读取固定887图，并与正式cache的VAL字节身份一致；重新在线提取，核对全部B/midpoint/candidate框、score、缺失/回退、GT、中心正确性/罚角与RIoU，以及完整汇总与所选epoch相符。
5. TEST读取固定1440图/原TEST字节身份，一次推理。每帧一次特征提取、三次native头调用（raw/original/raw-after）用于核验；实际回调计数887/2661或1440/4320，头/B state前后不变。GT只进入离线评价，在线调用不读GT/domain/sequence。
6. 指标沿用原完整视频protocol-v2，三个中心分母分别保存，纯周期角及90°罚角分开；GT转float32与正式缓存评价一致，不是改变原图坐标/尺度。VAL计算完整时序指标借用评价器test-mode，但console显式标VAL、JSON split为val，未访问TEST。
7. 终端仅打印overall的B与正式midpoint两张表；real/sim/各序列和raw candidate都在`*_compare.json`，不是多次独立推理。全部逐帧保存`*_rows.jsonl`，无新阈值、权重或架构搜索。

**本地检查事实：** 新增9项CPU检查与原正式12项检查均通过，覆盖完整formal selection重算、篡改/未完成拒绝、VAL/TEST静态隔离、无torch.load/CUDA/data、缺失/float32GT/分母以及VAL逐帧框/回退复核。原69项正式训练来源仍一致。真实GPU与完整887/1440在线评价待服务器运行；尚无正式midpoint性能可报告。

### 72.1 训练结束后的VAL结果

训练完成status必须是`FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED`，正式目录内已有：

- `selection.json`：按原VAL规则选中的头及SHA；
- `selected_val_compare.json`：所选头对B的完整VAL比较；
- `val_epoch_XX.json`及`val_epoch_XX.rows.jsonl`：各epoch汇总及887帧证据。

上述已经是VAL结果，无需重新训练或再次扫描。若要独立在线复核选中头，先静态后GPU：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_geometry_midpoint_formal_v1.py \
  --check-only --split val \
  --selection work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json \
  --out-dir work_dirs/port_geometry_midpoint_formal_v1_val_static

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_geometry_midpoint_formal_v1.py \
  --split val --gpu 0 \
  --selection work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json \
  --out-dir work_dirs/port_geometry_midpoint_formal_v1_val_eval
```

应输出`FROZEN_FORMAL_MIDPOINT_VAL_COMPLETE_REVIEW_REQUIRED`，保存`val_compare.json`、`val_rows.jsonl`、completion/progress/artifacts。若在线/缓存不一致，停止并回传failure，不忽略错误或重选epoch。

### 72.2 冻结正式头的TEST

用户已授权提供/执行该冻结TEST流程。头固定由原VAL选权，不要求把VAL所有诊断项都放宽为通过，也不据TEST修补系数或改选。TEST已多次暴露，本次仍只能称冻结方案在既有TEST的探索性报告，不称未接触确认。

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_geometry_midpoint_formal_v1.py \
  --check-only --split test \
  --selection work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json \
  --out-dir work_dirs/port_geometry_midpoint_formal_v1_test_static

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_geometry_midpoint_formal_v1.py \
  --split test --gpu 0 \
  --selection work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json \
  --out-dir work_dirs/port_geometry_midpoint_formal_v1_test_eval
```

应输出`FROZEN_FORMAL_MIDPOINT_TEST_COMPLETE_REVIEW_REQUIRED`。回传`test_compare.json`、`test_rows.jsonl`、completion/progress/artifacts。旧`eval_port_geometry_midpoint_v1_test.py`仍属于原200步诊断头，普通`tools/test.py`不能加载head-only `.pth`，不用于本正式头。

目录须新建；实际训练目录不同则统一修改selection路径，保留原文件。物理卡3映射逻辑0，运行环境及GPU型号需与正式缓存一致。结果压缩不包含源码、模型权重或大ROI缓存：

```bash
tar --exclude='*.pth' --exclude='*.pt' \
  -czf "work_dirs/port_geometry_midpoint_formal_v1_eval_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs \
  crane_symeood_k1_port_day2night_midpoint_formal_v1 \
  port_geometry_midpoint_formal_v1_val_static \
  port_geometry_midpoint_formal_v1_val_eval \
  port_geometry_midpoint_formal_v1_test_static \
  port_geometry_midpoint_formal_v1_test_eval
```

若未运行可选VAL复核，去掉对应两个val目录。此命令只生成服务器结果包，不在本地额外复制/解包结果，不改变可靠性另一对话的代码或记录。

## 73. 正式midpoint与有限训练的TEST比较（2026-10-04）

**已读取事实：** 用户回传正式评价终端全文，入口为`eval_port_geometry_midpoint_formal_v1.py --split test`，selection来自`work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json`，加载B的`crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth`，1440帧完成，状态为`FROZEN_FORMAL_MIDPOINT_TEST_COMPLETE_REVIEW_REQUIRED`。本次只读取终端，尚未收到正式selection/completion/test_compare/test_rows文件；选中头epoch、精确SHA、完整VAL表现、详细几何及分母计数仍待结果文件核验。不得把终端完成状态等同于已逐项复核上述文件。

**实验身份：** 有限训练为64个TRAIN身份、48 fit/16 probe、两尺度128视图、200步matched头；正式入口为同一midpoint结构和几何合同、新头初始化、全TRAIN 2558图的两尺度视图、24轮、原VAL规则选头。B始终冻结，正式训练不是端到端重训SymEOOD；正式头不是继续训练原短fit TEST头。正式入口完成状态要求完整24轮TRAIN/VAL及原选权合同通过，但本轮未直接审阅其训练结果文件。

下表为终端显示精度；B值与第68节/独立方法文档历史B逐项一致。正式入口同次重新推理B，未读取硬编码历史指标表。箭头表示期望方向。

| 指标 | B ep24 | 有限midpoint（200步） | 正式midpoint（VAL选头） | 正式相对有限 |
|---|---:|---:|---:|---|
| real全帧R_center % ↑ | 98.85 | 98.85 | 98.85 | 显示值不变 |
| real mean RIoU ↑ | 0.8169 | 0.8219 | 0.8193 | -0.0026，退化 |
| real DFR %/frame ↓ | 2.5106 | 2.6925 | 2.9857 | +0.2932，退化 |
| real ACI ↑ | 0.9440 | 0.9401 | 0.9424 | +0.0023，改善；仍低于B |
| real TDR_w10 % ↑ | 100.0 | 100.0 | 100.0 | 不变 |
| real MCML_max 帧 ↓ | 4 | 4 | 4 | 不变 |
| real MCML_mean 帧 ↓ | 3.5 | 4.0 | 3.5 | -0.5，恢复到B |
| real MCML≤5通过 | 1 | 1 | 1 | 不变 |
| real MRF 帧 ↓ | 2.29 | 2.67 | 2.33 | -0.34，改善；仍略高于B |
| sim协议A-RMSE ° ↓ | 1.8822 | 1.7884 | 1.2537 | -0.5347°，约降低29.9% |
| sim全帧R_center % ↑ | 100.0 | 100.0 | 100.0 | 显示值不变 |
| sim mean RIoU ↑ | 0.8815 | 0.8837 | 0.8943 | +0.0106，改善 |
| sim DFR %/frame ↓ | 2.4734 | 2.4707 | 2.6573 | +0.1866，退化 |
| sim ACI ↑ | 0.9479 | 0.9527 | 0.9606 | +0.0079，改善 |
| sim TDR_w10 % ↑ | 100.0 | 100.0 | 100.0 | 不变 |
| sim MCML_max/mean 帧 ↓ | 0/0.0 | 0/0.0 | 0/0.0 | 不变 |
| sim MCML≤5通过 | 1 | 1 | 1 | 不变 |

**测量结论：** 正式头相比有限头，sim协议角度、RIoU、ACI更好；real ACI、MCML_mean、MRF部分恢复，real平均RIoU更低，两域DFR更差。相比B，正式头两域RIoU、sim协议角度及ACI改善；两域DFR、real ACI/MRF有代价。不是所有指标提升，也不能把正式训练称为无效。最新已报告几何候选为SymEOOD+B＋正式midpoint；B仍为配对参考和冻结前端，未据TEST自动宣布全面替代，可靠性工作不自动切换前端。

**指标与证据限制：** R_center在此为原protocol-v2全帧指标，不是仅输出帧中心命中率；正式输出覆盖率、输出帧条件命中率和全帧正确覆盖的精确分子分母待JSON核对，不复制有限头计数充当正式实测。A-RMSE含无输出/中心≥10px的90°罚角；正式纯角RMSE及罚角计数未收到，不能用历史有限头的“sim无罚角”替正式头作证。DFR越低越好，数值升高是指标退化；它衡量对角线帧间相对变化，不足以独立证明错误抖动。ACI是角度连续性，不能代替真值角度精度。两次汇总精度有限，未做统计显著性检验。

**机制推断：** 更多TRAIN监督可能增强sim几何泛化；两域DFR升高可能涉及尺寸修正的逐帧变化、真实尺度变化或修正/回退切换，尚未定位根因。两次数据量、步数及选权方式共同变化，不能把差异单独归于数据量。

**待验证与下一步：** 优先收到正式selection/completion、selected_val_compare、test_compare/test_rows，核验身份及完整VAL，并复用逐帧预测检查预测/GT对角线变化及回退切换，不先新增训练。后续时序优化的设计/系数/选权限定TRAIN/VAL；TEST已多次暴露，不据此次结果调参、重选或宣称未接触确认。本轮只更新文本记录，没有改模型/训练/评价代码、运行训练或连接服务器。


## 74. 正式midpoint全量结果包的VAL、选权和指标核验（2026-10-04）

**证据事实：** 收到`port_geometry_midpoint_formal_v1_full_review_20261004_090841_808764.tar.gz`，SHA256=`a141189dbf02142b95e46c646f3bdbbba391b6d2399c2e7ba54e402096930746`。68个成员，未压缩内容80,329,472字节，含正式24轮VAL逐帧与汇总、基线、选权、TRAIN进度，以及VAL/TEST独立在线复核、静态completion及artifacts。直接内存读取，没有在项目复制/解包JSON、权重或一次性审查脚本。

### 74.1 身份与计算正确性

- 63个包内被artifacts索引的文件SHA均一致；训练69项、评价73项来源逐项与当前源码一致，训练completion中的来源/protocol身份另核对通过。24个权重未包含，不能在本机独立复算实际权重文件SHA；服务器成功入口已做权重/状态校验，包内selection与VAL/TEST记录相互一致。
- 正式训练24轮、每轮902步，共21,648次更新；21,648条TRAIN损失/梯度均有限，更新序号、epoch日志及selection元数据一致。B更新0；5116个TRAIN视图中5103个符合原B中心<15px监督资格（4缺输出、9中心不匹配），不是每个视图都提供梯度。
- 原选权规则从24组完整VAL指标重算，仍选`head_epoch_23.pth`，23轮20,746次更新，记录权重SHA为`2c4c5ae9e071cbdb13fb722f4b2d3986861c62bd60c53568c72fb9b02d662f0b`。所有24头均满足原可行域，epoch23软评分最高；未用TEST选权。原评分包含sim ACI/角度及中心/窗口命中，不包含DFR、尺寸误差或平均RIoU，故选中头与几何联合检查不全部通过可以同时成立。
- 零修正初始VAL的B/midpoint汇总相同；24轮B逐帧框、GT、指标均相同。B身份仍为原尺度增强B ep24，config及checkpoint标识一致，TEST显示值与历史B一致。
- 每轮VAL887帧（real375、sim512），TEST1440帧（real868、sim572），无重复身份。独立在线VAL逐帧框/分数/GT/回退/派生指标与选中epoch23相同，汇总一致；VAL/TEST模型参数和buffer state前后不变，实际提取/头调用887/2661与1440/4320，更新0。
- 对全部24轮VAL＋独立在线VAL＋TEST共23,615条帧记录重新汇总全部分组、B/midpoint/candidate指标、失败段和配对计数；逐框重新计算中心、长短边、周期角度及90°罚角，未发现不一致，最大数值差约3.22e-15。输出数量/score不变，整框回退精确等于B，接受候选与交付框一致。
- 原OpenCV RIoU采用包内逐帧值重汇总；另对选中在线VAL/TEST共6951个有效B/midpoint/candidate框，用独立float64凸多边形裁剪交叉检查。VAL最大RIoU差1.57e-5，TEST最大1.37e-4，平均绝对差分别5.66e-7/7.39e-7，所有RIoU≥0.5判定一致。当前本机未装cv2，未重跑服务器原OpenCV算子；这些差异属于两种几何计算的数值差，保留原协议数值，不修改报告。

**核验边界：** 未重新GPU推理、重新训练或独立读取模型权重；实际图像/标注字节身份依赖成功服务器入口的校验记录，本轮不重审数据划分。记录的是结果包完整性、计算一致性和独立几何交叉证据，不是独立重复训练或统计显著性验证。

### 74.2 完整VAL主指标（epoch23，B→正式midpoint）

| 指标 | real | sim |
|---|---|---|
| 全帧R_center % ↑ | 96.00→96.53 | 100→100 |
| 全帧mean RIoU ↑ | 0.7955→0.7918 | 0.8857→0.8930 |
| 协议A-RMSE ° ↓ | 原主表不输出 | 2.1072→1.5656 |
| DFR %/frame ↓ | 4.5430→4.6580 | 2.4130→2.5419 |
| ACI ↑ | 0.9369→0.9418 | 0.9499→0.9578 |
| TDR_w10 % ↑ | 100→100 | 100→100 |
| MCML_max/mean 帧 ↓ | 4/2.0→4/2.0 | 0/0.0→0/0.0 |
| MCML≤5通过 | 1→1 | 1→1 |
| MRF 帧 ↓ | 1.57→1.57 | 无失败恢复事件，未输出 |

real_seq07 RIoU 0.737034→0.732909；real_seq14 0.884194→0.881134；sim_seq10 0.885695→0.892994。real两个序列都略退化，不能把real聚合退化解释为仅分组权重变化。VAL real恢复2个中心正确输出、无新增中心失败；两域无新增RIoU<0.5失败，最长中心/RIoU失败段不变。25项联合检查19项通过，6项失败：real RIoU、real短边、两域DFR、sim长边和短边；原选权不改变，未自动晋升。

### 74.3 三个中心分母（阈值<15px）

| 分组 | 输出覆盖 B→正式 | 输出帧中心正确 B→正式 | 全帧中心正确覆盖 B→正式 |
|---|---|---|---|
| VAL real | 374/375→374/375（99.7333%） | 360/374→362/374（96.2567%→96.7914%） | 360/375→362/375（96%→96.5333%） |
| VAL sim | 512/512→512/512 | 512/512→512/512 | 512/512→512/512 |
| TEST real | 859/868→859/868（98.9631%） | 858/859→858/859（99.8836%） | 858/868→858/868（98.8479%） |
| TEST sim | 572/572→572/572 | 572/572→572/572 | 572/572→572/572 |

sim各项均100%。此处已由正式逐帧结果核验，不再仅引用有限头计数。原终端R_center属于第三列，不能称为第二列的输出帧命中率。

### 74.4 细几何与纯角度（只统计输出帧，B→正式）

| 指标 | VAL real | VAL sim | TEST real | TEST sim |
|---|---|---|---|---|
| 中心mean px ↓ | 9.5746→9.2424 | 1.6452→1.2742 | 2.6628→2.2309 | 1.1452→0.8490 |
| 中心RMSE px ↓ | 34.5779→34.4576 | 2.0332→1.6108 | 3.1767→2.8778 | 1.3645→0.9955 |
| 长边相对误差mean % ↓ | 8.4208→8.3865 | 3.8712→4.5522 | 5.1917→5.2950 | 4.0338→4.1753 |
| 短边相对误差mean % ↓ | 8.7580→9.2394 | 4.1388→4.3488 | 6.5412→8.4585 | 4.4780→4.6671 |
| 纯周期角RMSE ° ↓ | 3.7700→3.5703 | 2.1072→1.5656 | 2.6210→2.6881 | 1.8822→1.2537 |

sim在VAL/TEST均无缺输出或中心≥10px罚角，故协议A-RMSE与纯角RMSE一致：先前终端阶段无法确认的纯角度收益现在得到验证。real VAL缺输出罚角1→1、中心≥10px罚角28→26；TEST缺输出9→9、中心罚角3→5。real TEST全帧含罚角RMSE 10.8867→11.7221，纯角RMSE 2.6210→2.6881；不能把含罚角变化全部归因于方向精度。两个新增≥10px事件仍<15px，故可与全帧中心正确数不变同时发生。

**对上一轮结论的补充：** 正式头的两域平均定位和sim纯角度收益得到确认，但“主要只有DFR有代价”不完整：尺寸误差多数退化，尤其TEST real短边6.54%→8.46%；VAL real RIoU退化，TEST real纯角RMSE/ACI略退化。正式TEST三个序列RIoU仍都改善，终端数值与JSON一致；无需撤销已测收益，也不能用RIoU改善代表尺寸已改善。VAL与TEST方向有差异，不代表评价计算错误。

### 74.5 下一步范围

**建议，尚未实施：** 优先复用已保存VAL逐帧框，核对短边/对角线的有符号残差、预测与GT的相邻帧变化、分序列与原B尺度；把静态尺寸偏差和时序尺寸变化一起检查。VAL两域回退均0，故其DFR退化不能归因于交付框在候选/B间切换；TEST real仅5个回退、sim为0，回退不是两域共同代价的充分解释。不能据此断言唯一根因，也不先通过平滑掩盖系统尺寸偏差。保持等比例变换、原图坐标还原、原深度估计接口；未测深度精度。后续新增监督/系数/选权只在TRAIN/VAL确定，已多次暴露TEST只用于冻结报告。本轮仅更新现有文本记录，无代码修改、训练、服务器连接或可靠性工作线改动。


## 75. 用户授权的VAL短边/对角线残差核对入口与近年文献（2026-10-04）

**授权与实现事实：** 用户要求按第74.5节核对短边和对角线残差，修改对应代码、复核后给服务器指令，并询问2024/2025/2026相关论文。本轮新增[只读诊断入口](../crane_project/tools/analyze_port_geometry_midpoint_size_temporal_v1.py)及[15项检查](../tests/test_port_geometry_midpoint_size_temporal_v1.py)。只使用标准库，兼容Python3.8；不加载模型/ROI缓存/图片，不导入torch/mmcv/CUDA，不训练、滤波、改框、调参或访问TEST。原formal训练/评价源码、69/73来源和协议不改；另一个可靠性工作线的既有修改保留。本轮未连接服务器。

### 75.1 输入、指标及边界

1. 固定读取正式目录的completion/selection/artifacts、选中epoch的VAL逐帧与汇总和24组VAL摘要；重新核对已保存原选权一致性，不根据新残差改选、不读取其他epoch逐帧框或权重。要求固定887帧VAL序列，禁止把TEST逐帧当输入。
2. 核对原69项源码、协议和B身份、结果SHA、原选权、逐帧正尺寸/分数/输出/回退、中心/周期角/长短边派生指标及三个分母。可选独立在线VAL目录进一步要求同一selection、completion字节SHA和选中头state、前后状态不变、887/2661提取/调用、GT/框/回退/派生指标逐帧精确一致。
3. 静态分别计算长边、短边、对角线的像素残差、带符号相对误差`P/G-1`、`log(P/G)`、偏差/MAE/RMSE/标准差/分位数和midpoint/B修正。负号表示框偏小；只在离线指标中规范化长短边和π周期，原交付框及坐标不变。
4. 时序只配对同序列连续frame_id且前后两帧均有B/正式输出；缺输出或帧号缺口断开。两方法与GT共用相同配对支持。另报排除计数、GT对角线变化和原DFR/ACI复算；角度变化误差按π周期处理，ACI沿用35°合同。
5. 核心GT相对残差为`delta log(P)-delta log(G)`，等于相邻帧静态log尺寸误差之差；midpoint与B的此残差之差又等于`delta log(midpoint/B)`。该分解用于区分新增修正变化与GT标注尺度变化，不能据单个均值断言根因。另报普通相对增长差、相关系数和角度增长误差，恒定序列相关系数为null。
6. real/sim/每序列及overall报告；按原图GT/B短边16/24/32/48px分箱，只是描述分组，不是新的训练/接受阈值。另按既有15px中心合同报告B中心正确/错误和双方中心正确子集，保留全部输出主结果，避免错误中心框掩盖尺寸证据。
7. 原DFR/ACI复算必须与保存协议四位显示值一致；不重算OpenCV RIoU或重新审计数据集。GT运动为OBB标注变化，未证明物理尺度/深度精度；残差差异/相关系数不等于统计显著或唯一因果。

输出为`summary.json`、`static_frames.jsonl`、`temporal_pairs.jsonl`、`completion.json`、`artifacts.json`；静态检查仅写completion/artifacts。要求新输出目录，拒绝覆盖。正常状态为`VAL_SIZE_TEMPORAL_RESIDUALS_COMPLETE_REVIEW_REQUIRED`；静态为`STATIC_VAL_RESIDUAL_INPUTS_PASS_NO_GPU_NO_UPDATES`。头/检测器更新及推理调用均0，selection_on_test和automatic_promotion均false。

### 75.2 本地必要验证与已有VAL诊断事实

15项CPU检查全部通过，覆盖同等框宽高交换/角度等价及等比例缩放、DFR降低却GT跟随退化、恒定尺寸偏差与动态残差分离、缺输出/间隙/序列边界、角度周期、空支持与恒定相关、输入不变、SHA篡改、选权篡改、错误分母/score/尺度/TEST序列拒绝、独立在线VAL身份/重放、静态不执行分析及拒绝覆盖。两个新文件通过Python3.8语法及内存编译，未重新运行无关模型检查。

复用第74节原压缩包、直接内存读取选中epoch23和独立在线VAL：新诊断输入合同、分析、JSON/JSONL序列化、行数和输出artifacts SHA均验证通过。测试输出只在自动清理的临时目录中，未保存在项目；没有重新推理或产生新模型性能。887帧形成884个连续身份对，排除2个缺输出邻接对，实际882对（real371、sim511），两个序列边界均断开。

| 已保存VAL的派生指标：B→正式 | real | sim |
|---|---|---|
| 短边相对误差signed mean % | -5.0762→-5.3352 | -3.0513→-2.9849 |
| 短边相对误差MAE % | 8.7580→9.2394 | 4.1388→4.3488 |
| 短边log残差std | 0.102285→0.106378 | 0.038892→0.043526 |
| 短边GT相对log变化残差RMSE | 0.094100→0.097720 | 0.027866→0.036599 |
| 对角线GT相对log变化残差RMSE | 0.086179→0.082685 | 0.028971→0.031190 |
| GT对角线DFR，同配对 %/frame | 2.524855 | 1.012393 |
| 原预测DFR %/frame | 4.5430→4.6580 | 2.4130→2.5419 |
| GT相对角度变化误差RMSE ° | 3.055144→2.854156 | 1.908536→1.415072 |

两个real序列短边动态残差均略退化，对角线动态残差均略改善；real_seq07主要偏小（signed mean -9.1104%→-9.5906%），real_seq14略偏大（1.0156%→1.0908%），不是统一固定缩放偏差。仅双方中心<15px的real360帧中，短边MAE仍7.9064%→8.3784%；不是只有错误中心框产生尺寸代价。sim512帧短边均值偏差略改善但分散与帧间变化残差增大，不能把其退化全解释为恒定偏小。

**机制推断与下一步：** 优先关注四点/矩形读出后的短边误差与sim逐帧尺寸修正变化；real不能仅以DFR上升认定GT跟随变差。不先引入统一尺度补偿或单纯压制变化的平滑，不将以上相关性宣称根因。新的尺寸/时序监督仍待本轮诊断审查后固定；不放行新训练，不改epoch23或接受阈值，TEST已多次暴露但本次入口不读TEST。保持原等比例坐标和深度接口；未验证深度误差。

### 75.3 服务器复现与结果包

只需按相对路径上传新入口；检查文件可一并上传，不更新旧manifest。CPU标准库执行，没有显卡数量/显存需求。首先执行静态合同：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/analyze_port_geometry_midpoint_size_temporal_v1.py \
  --check-only \
  --training-dir work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1 \
  --val-eval-dir work_dirs/port_geometry_midpoint_formal_v1_val_eval \
  --out-dir work_dirs/port_geometry_midpoint_size_temporal_v1_static
```

静态状态通过后运行同一已选头的完整VAL残差分析：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/analyze_port_geometry_midpoint_size_temporal_v1.py \
  --training-dir work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1 \
  --val-eval-dir work_dirs/port_geometry_midpoint_formal_v1_val_eval \
  --out-dir work_dirs/port_geometry_midpoint_size_temporal_v1_val
```

如输出目录已存在，更换输出后缀；不删除已有结果。本次独立在线VAL已回传，命令使用它；未完成在线VAL的其他环境可省略`--val-eval-dir`，仍只分析正式保存VAL，报告会明确没有额外在线重放证据。

```bash
tar -czf "work_dirs/port_geometry_midpoint_size_temporal_v1_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  -C work_dirs \
  port_geometry_midpoint_size_temporal_v1_static \
  port_geometry_midpoint_size_temporal_v1_val
```

仅压缩新诊断文本结果，无模型、ROI缓存或源码。若相同输入/源码，服务器应重现本地887帧/882对及上述派生数值；服务器此次运行尚待回传，不能写成已执行。

### 75.4 2024—2026相关原始文献补充

本轮按原出版社、会议论文或作者arXiv核对日期/摘要，并区分可借鉴方向与已验证机制。属于针对性检索，未证明当前组合全球首次，也未在本项目实现这些论文模块。

| 文献与已核验年份 | 相关内容 | 对当前midpoint的借鉴与限制 |
|---|---|---|
| [Oriented R-CNN and Beyond，IJCV2024，2024-01-29出版](https://link.springer.com/article/10.1007/s11263-024-01989-w) | midpoint-offset旋转候选、ROI精修及实例分割扩展 | 可用于最新相关工作/编码对照；其外接水平框midpoint offset不是本项目四个OBB边中点，不将两者等同 |
| [Rethinking Boundary Discontinuity Problem for Oriented Object Detection，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Rethinking_Boundary_Discontinuity_Problem_for_Oriented_Object_Detection_CVPR_2024_paper.html) | 区分角度编码可逆性与联合优化，指出仅平滑IoU类损失不保证预测边界连续 | 支持同时检查编码/解码及纯角/角度连续性；不是本项目DFR问题根因证据 |
| [D-FINE，ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html) | 以细粒度概率分布逐层精修定位，另含定位自蒸馏模块 | 可借鉴分布表示与最终定位质量的联系；不是四中点OBB或视频DFR方案，不引入其蒸馏/替换前端 |
| [Strip R-CNN，2025-01-07作者预印本](https://arxiv.org/abs/2501.03775)，[AAAI2026正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/38217) | 正交条带卷积与定位分支，处理细长目标的方向性空间信息 | 可借鉴长短轴方向信息不对称的诊断；现有9×9头换条带是否有益未验证，本轮不修改结构。避免把2025预印本误写为AAAI2025 |
| [RiO-DETR，2026作者论文v2，2026-07-30；页面标注ECCV2026接收](https://arxiv.org/abs/2603.09411) | 内容驱动方向估计、周期性的有界粗到细精修 | 可借鉴周期更新/解耦精修的评价思路，不替换SymEOOD或引入新分配策略；论文未验证本项目四点校准或DFR收益 |

四边中点最直接先例仍为[MidNet 2021](https://arxiv.org/abs/2111.10961)，不能用近年论文替代对它的讨论。当前期望差校准仅保证指定先验的零修正中性，不证明一般积分偏差消除。论文主张应围绕冻结高覆盖前端下的局部几何纠偏及其精度/连续性证据，公平消融先验校准与点读出，不能靠新引用或指标微调自动增加创新性。
