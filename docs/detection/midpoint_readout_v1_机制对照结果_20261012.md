# midpoint 同容量读出机制对照：正式训练与结果记录

日期：2026-10-12（Asia/Taipei）。当前状态：**24轮正式训练、完整 VAL 与固定 TEST 对照均已完成，本机核验通过。支持保留原高斯先验；不替换既有σ1.5版本。**

**固定TEST补充（已完成）：** A067在GPU2完成两头887帧在线VAL门槛及1440帧固定TEST，退出0，本机逐帧与摘要复算通过。均匀epoch04／σ1.5-epoch03及B24不重选，旧B／高斯TEST结果复现；无训练或自动替换。下文VAL结果保留原身份，TEST表与反面结果另列于末尾，可作为第五节同集合的机制对照。

<a id="deepseek-writing-summary"></a>

## DeepSeek 优先读取：对照内容与时序结果摘要

**实验问题：** B框高斯空间先验是否较均匀先验带来额外收益？固定B检测器epoch24，将中性残差读出中的高斯先验替换为均匀先验；保留17,696参数、ROI／网络、共同初值、监督、优化器、24轮预算、VAL选权、坐标还原和接受／回退。比较B、均匀头epoch04、高斯σ1.5头epoch03。两头都完成24轮训练，epoch04／03是完整VAL选中的快照。评价固定1440帧TEST，real868＋sim572；TEST没有训练、选参或重选权。

**证据入口：** 正文表格使用下文“固定 TEST 对照”的数值；前面的VAL表只作选权／开发依据，不能拼入同一TEST表。直接来源为[固定TEST六行报告](midpoint_readout_fixed_test_evidence_20261012/test/compare.json)，逐序列MCML来自两份完整summary。所有数值已核对，不以终端截图或旧σ=1结果替换σ=1.5。

**DFR和MCML：** DFR数值越低越好；当前高斯头real DFR较B和均匀头更高，表现退化，sim DFR更低，表现改善。MCML最大值与MCML均值必须分开：最大值real都为4帧、sim都为0，均通过5帧门槛；real MCML均值高斯4.0帧，高于B3.5和均匀3.0，存在代价，不能只写“MCML不变”。

| TEST指标（↓越低越好） | B | 均匀先验 | 高斯σ1.5 | 高斯相对B |
|---|---:|---:|---:|---|
| real DFR(%/frame) | 2.5106 | 2.6757 | 2.7019 | 退化 |
| sim DFR(%/frame) | 2.4734 | 2.3051 | 2.2299 | 改善 |
| real MCML_max(帧) | 4 | 4 | 4 | 不变 |
| real MCML_mean(帧) | 3.5 | 3.0 | 4.0 | 退化 |
| sim MCML_max(帧) | 0 | 0 | 0 | 不变 |
| sim MCML_mean(帧) | 0.0 | 0.0 | 0.0 | 不变 |
| real MRF(帧) | 2.29 | 2.14 | 2.43 | 退化 |

MCML在本协议下依据“有输出且RIoU≥0.5”的命中标志：无输出或RIoU<0.5均计为失败。每个连续视频段计算最长连续失败；MCML_max取这些段最大值，MCML_mean取这些段最长失败长度的算术平均，**不是所有失败事件时长的均值，也不是纯缺输出的连续长度**。因此，输出数与中心命中完全相同，MCML_mean仍可变化。real_seq03三个条件的最长连续失败均4帧；real_seq04为B3／均匀2／高斯4帧，故总体均值分别(4+3)/2=3.5、(4+2)/2=3.0、(4+4)/2=4.0。MRF另衡量已恢复失败事件的平均恢复帧数，不与MCML_mean混同。

DFR是相邻连续且有输出帧间的预测框对角线相对变化率，公式为`100×mean(|d_t−d_{t−1}|/d_{t−1})`；缺输出或帧号间断不跨越计算。它不是漏检率，也不是已扣除GT真实运动的抖动残差。本文只按既有协议讨论DFR高低，不将单一DFR改善等同于物理控制或深度收益。

**可写结论：** 高斯空间先验在同容量、同预算的固定TEST对照中提高两域RIoU、sim角度精度及两域ACI，中心命中和输出覆盖保持；real DFR、MCML_mean、MRF及部分尺寸误差有代价。不能写“所有连续性／系统级指标保持或提高”，也不能只保留MCML_max忽略均值。此测试复现了既有σ1.5性能，没有通过TEST再微调或替换版本。

**写作限定：** 小论文采用一张精简机制消融表和一段结果解释；细分尺寸／候选与来源核验可放附录。保留单种子、TEST多次暴露、σ1.5采用发生于旧TEST回传后的事实；对照未单独隔离中性期望相减，不能写成根因证明、显著／稳定泛化或先验必不可少。可靠性、深度、控制内容不改，不填本轮未计算的均匀头AP。本文档为文字参考，论文正文尚未修改。

## 固定对照与执行身份

- B：已封存的检测器 epoch_24。
- 高斯先验：已完成的 σ=1.5／head_epoch_03；由原全部24轮完整 VAL 选出，本轮复用，不重新训练或选权。
- 新对照：均匀先验中性残差读出 `a + E softmax(L) − E softmax(0)`，与高斯先验 `a + E softmax(Q+L) − E softmax(Q)`比较。
- 保留17,696参数、9×9 ROI、259通道、共同初值、seed1703、点监督／SmoothL1、Adam学习率0.001、clip10、batch8（4real＋4sim）、原坐标还原／解码／接受与整框回退。只新训均匀头，902步×24轮＝21,648更新，全部24轮完整标准尺度 VAL 按原规则选权。
- 此对照检验保留中性残差形式时B高斯空间先验的整体作用；分布、梯度和可达残差范围随之改变，不能单独证明期望相减项、σ最优、根因或统计显著性。

## 服务器运行

用户明确授权正式训练；GPU2与GPU3启动前均空闲，选择物理GPU2／逻辑0，GTX1080。工作目录 `/media/omnisky/personal_files/ljj/symEOOD`；环境 `mmrotljj`；临时设置 `LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64`，匹配原cuDNN8801。当前87份来源与通过的预检一致，实现提交 `cfaf168d11967326debcd9fae06cb03523f1d516`，服务器启动时HEAD记录在run_metadata.json，不强制回退其它窗口的新提交。

运行目录：`work_dirs/port_geometry_midpoint_readout_v1/formal_20261012_gpu2_v1/`。后台启动器PID421632，使用nohup，训练输出在`train/`；日志`run.log`、退出码`exit_code.txt`、实际启动环境与命令`run_metadata.json`。PID仅为启动时身份，不作为完成依据。服务器原件不覆盖。

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/train_port_geometry_midpoint_readout_v1.py \
  --stage train --gpu 0 \
  --preflight-report work_dirs/port_geometry_midpoint_readout_v1/run_cfaf168_20261011_02/preflight/completion.json \
  --out-dir work_dirs/port_geometry_midpoint_readout_v1/formal_20261012_gpu2_v1/train
```

命令在激活mmrotljj及上述临时库路径后执行。B不构建／forward／更新，不重建缓存，不读取TEST，不改变既有模型或自动采用新头。

## 核验要求与论文边界（已执行）

完成依据：退出码0、`READOUT_FULL_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED`、24轮及21,648更新、选权／重载完整 VAL 一致。随后核验原参考重放、来源／缓存／共同初值、全部24轮选择记录、887帧配对结果，以及输出帧中心命中率、输出覆盖率、全帧中心正确覆盖、RIoU、sim角度、DFR、ACI、TDR、MCML、MRF和接受／回退计数。

本次正式结果见下文；零更新预检仍只保留工程验证身份。TEST已多次暴露，σ1.5工程采用发生在旧TEST回传后；本轮TRAIN／VAL对照不消除该限制。可靠性、深度与论文原稿不改。

预检依据：[已通过的零更新预检](midpoint_readout_evidence_20261011/README.md)。集中活动记录：[A066](../geometry_precision_activity_20261005.md#activity-a066)。

## 正式完成事实与收件核验

- 服务器状态`READOUT_FULL_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED`、退出码0，24轮、21,648次更新，耗时1,020.995秒（约17.02分钟，含合同／缓存检查、参考重放、训练、每轮VAL、保存及最终复核；不是完整检测FPS）。21,648条optimizer_step与21,648条train记录按每轮902步顺序严格一致，24条epoch_complete齐备，梯度与损失均有限。
- 全部24轮使用既定完整VAL规则选权，均为可行候选；选中均匀头`head_epoch_04.pth`（该快照3,608更新），并不表示整项实验只训练4轮。权重SHA256 `d0ec9a60ee33b0843128d65e5a3127291d5948926e14128bfbe7b8d07acb0c05`，保存恢复与选中完整VAL重算一致。原高斯头仍为σ1.5／head_epoch_03（其完整训练亦24轮）。
- 冻结B forward／更新／新特征提取均0，CPU缓存加载3份，TEST访问false。采用共同初值，不从预检权重或高斯epoch03续训。
- PyTorch allocated峰值2,117,120字节＝2.019 MiB，reserved峰值4,194,304字节＝4 MiB。只指该缓存小头正式运行的分配器峰值；运行中nvidia-smi快照约214 MiB包含进程额外占用，不能替代完整前端推理显存。
- 服务器逐份核对全部artifact文件SHA及完整优化轨迹。本机核对7份索引内必要报告／逐帧文件SHA、预检proof／共同初值／原参考身份；独立复算全部24轮选权，887帧（real375＋sim512）的2563个摘要数值最大差0。逐框重新分解B／均匀头／候选几何，最大差6.7363e−6（≤1e−5容差）。没有本机模型推理或权重重载；权重只留服务器。
- 回传包808,190字节，SHA256 `8db5cb0731cca023197cb25bb0fd99e1418c7dca6ffacac2d9ca0ef7a05d58fd`。只保留重要报告、选中轮次逐帧VAL、日志和运行／收件记录；未复制全部中间结果、权重、缓存或源码。本机与服务器传输压缩包已删除并核验不存在，清理状态见本机receipt；服务器完整24轮原件保留。

## 完整标准尺度 VAL 对照

以下均为VAL，不与旧TEST指标混写。RIoU／DFR／ACI取既有full_video_protocol_v2口径；RIoU包含缺输出帧的0，表内仅保留原报告4位小数。sim A-RMSE为协议角度项，本组全部sim帧输出且中心有效，其值与纯周期角度RMSE四舍五入后一致。real不报告角度真值精度。

| 域 | 条件（VAL选中轮次） | RIoU ↑ | sim A-RMSE(°) ↓ | DFR(%/frame) ↓ | ACI ↑ | MCML_max(帧) ↓ |
|---|---|---:|---:|---:|---:|---:|
| real | B／检测器24 | 0.7955 | — | 4.5430 | 0.9369 | 4 |
| sim | B／检测器24 | 0.8857 | 2.1072 | 2.4130 | 0.9499 | 0 |
| real | 均匀先验／头04 | 0.7957 | — | 4.6147 | 0.9379 | 4 |
| sim | 均匀先验／头04 | 0.8869 | 1.7298 | 2.3669 | 0.9548 | 0 |
| real | 高斯先验σ1.5／头03 | 0.7966 | — | 4.5394 | 0.9449 | 4 |
| sim | 高斯先验σ1.5／头03 | 0.8914 | 1.5635 | 2.3521 | 0.9600 | 0 |

### 中心与输出覆盖（三个分母分开）

| 域 | 条件 | 输出帧中心命中率 ↑ | 输出覆盖率 ↑ | 全帧中心正确覆盖 ↑ |
|---|---|---:|---:|---:|
| real | B／检测器24 | 360/374＝96.2567% | 374/375＝99.7333% | 360/375＝96.0000% |
| sim | B／检测器24 | 512/512＝100.0000% | 512/512＝100.0000% | 512/512＝100.0000% |
| real | 均匀先验／头04 | 361/374＝96.5241% | 374/375＝99.7333% | 361/375＝96.2667% |
| sim | 均匀先验／头04 | 512/512＝100.0000% | 512/512＝100.0000% | 512/512＝100.0000% |
| real | 高斯先验σ1.5／头03 | 362/374＝96.7914% | 374/375＝99.7333% | 362/375＝96.5333% |
| sim | 高斯先验σ1.5／头03 | 512/512＝100.0000% | 512/512＝100.0000% | 512/512＝100.0000% |

上述命中以既有15px中心阈值统计输出帧；不直接沿用包含全帧分母的历史R_center显示值。三条件所有域TDR_w10均100%；real MCML_mean均2.0帧／MRF均1.57帧，sim MCML_mean均0，MCML_pass均1。两种头所有输出框均接受（real374、sim512），回退与无效候选均0，缺输出的1帧不算回退；改善不是回退掩盖的结果。

### 独立尺寸与中心误差：不作全面改善声明

下表仅统计输出帧；长／短边是几何长短轴排序后的相对绝对误差均值，不替代原图在线w／h的轴关联。real无可靠角度真值，尺寸／中心仍可评价。

| 域 | 条件 | 中心误差均值(px) ↓ | 长边相对误差均值 ↓ | 短边相对误差均值 ↓ |
|---|---|---:|---:|---:|
| real | B／检测器24 | 9.574623 | 0.084208 | 0.087580 |
| sim | B／检测器24 | 1.645235 | 0.038712 | 0.041388 |
| real | 均匀先验／头04 | 9.465023 | 0.084488 | 0.083729 |
| sim | 均匀先验／头04 | 1.830998 | 0.039066 | 0.039795 |
| real | 高斯先验σ1.5／头03 | 9.653240 | 0.082202 | 0.084058 |
| sim | 高斯先验σ1.5／头03 | 1.440763 | 0.043216 | 0.043450 |

均匀头相对B恢复2帧real中心，同时新增1帧失败，净增1；高斯头恢复2帧且新增失败0，净增2。均匀头新增失败帧`real_seq07_00146`：中心误差13.4976→17.7884px，跨越15px阈值。两头均无新增RIoU低于0.5的帧，sim无新增中心失败。

## 分析：事实、推断与待验证

**事实。** 在同容量、共同初值、同预算和同选权规则下，原高斯头相对均匀头：real RIoU高0.0009、DFR低0.0753个百分点／帧、ACI高0.0070，中心正确多1帧；sim RIoU高0.0045、A-RMSE低0.1663°、DFR低0.0148个百分点／帧、ACI高0.0052。两域覆盖不变，所有候选均接受。均匀头相对B的sim角度／RIoU／连续性也有收益，说明收益不能全部归于高斯先验。

**必须同时保留的代价。** 均匀头的real平均中心误差与短边相对误差更低，sim长／短边相对误差均更低；高斯头不是全部独立尺寸指标最优。高斯头sim长／短边相对误差亦高于B，不应写成尺寸、中心均值、方向及时序全指标全面改善。综合RIoU与单独尺寸误差衡量不同的几何因素。

**推断。** 本次结果与“B空间先验约束四点概率读出，在定位、方向及连续性之间提供更好的综合折中”一致；可以作为机制支持。先验改变概率分布、梯度和可达残差范围，不能进一步单独归因于某项正则化或先验期望相减，也不能据此确定原几何误差的根因。

**待验证／边界。** 单种子、已知VAL、复用旧高斯训练；没有新独立序列、置信区间或统计显著性，不证明必要性、跨任务普适性、深度或可靠性收益。TEST已多次暴露，σ1.5工程采用时序不改。本轮不访问TEST、不新增AP、不修改原论文／可靠性／深度，不自动推广新均匀头。

## 下一步与论文写作范围

保留原B＋高斯midpoint σ1.5／head_epoch03，均匀头作为机制对照封存。**本项正式机制对照已完成，可以补入论文的VAL消融表及简短分析**；写成“在本数据与评价协议下，高斯空间先验改善综合几何与连续性表现”，同时保留独立尺寸误差的反面结果。不要写成首次提出中点／积分回归、证明先验必不可少、所有指标改善或显著／稳定泛化。当前不需要为本项再启动参数扫描或TEST选权；外部检测器与独立序列验证仍是各自的投稿补强项目。

可直接参考的简短表述：

> 在相同冻结检测器、头容量、训练预算与VAL选权规则下，将B框高斯空间先验替换为均匀先验。σ=1.5的高斯读出在两域RIoU、DFR和ACI及仿真角度误差上优于均匀读出，并保持输出覆盖；真实域中心正确帧数增加1帧。均匀读出在部分独立尺寸误差上更低，表明先验的作用主要体现为本协议下定位、方向与连续性的综合折中。本结果为单次TRAIN／VAL对照，不代表所有几何分量全面改善。

## 重要结果入口

- [六行完整对照及全部24轮摘要](midpoint_readout_fixed_test_evidence_20261012/train/val_readout_compare.json)
- [完整24轮选权与权重身份](midpoint_readout_fixed_test_evidence_20261012/train/selection.json)
- [正式完成／运行来源](midpoint_readout_fixed_test_evidence_20261012/train/completion.json)
- [均匀头选中轮次887帧原始预测](midpoint_readout_fixed_test_evidence_20261012/train/val_epoch_04.rows.jsonl)
- [本机收件与独立复算](midpoint_readout_fixed_test_evidence_20261012/train/local_review_receipt.json)

服务器完整原件及24份权重：`/media/omnisky/personal_files/ljj/symEOOD/work_dirs/port_geometry_midpoint_readout_v1/formal_20261012_gpu2_v1/train/`。本地保留必要报告，不归档全部中间文件；原artifact列出的其它文件仍留服务器，不能把当前子集当作完整训练包。

## 固定 TEST 对照（A067，2026-10-12）

### 运行与验证事实

新增独立入口`crane_project/tools/eval_port_geometry_midpoint_readout_test_v1.py`（check／val／test三阶段）、独立协议与91份当前源码清单及21项检查；原训练87份清单／源码、旧头和历史清单不改。本机新增21项与原22项合计43通过、Python3.8语法／来源／diff通过；服务器mmrotljj下新增21项全部通过，静态核对无TEST读取／tensor反序列化／GPU调用。

本轮4份新增代码直接同步至服务器对应tools／tests，未覆盖已有文件或拉取其它窗口改动，未Git提交／推送。GPU2空闲、GPU3已有进程，选择GPU2／逻辑0。环境Torch1.13.1+cu117、CUDA11.7、cuDNN8801、MMCV1.7.0、MMDetection2.25.1、MMRotate0.3.4、OpenCV4.13.0、GTX1080。新清单绑定当前5份检测类字节；这里实际执行检测器，不能沿用训练阶段“未执行”的豁免。冻结B状态等于原特征提取状态，再以两个头完整887帧在线VAL验证B、候选／接受／回退、原始轴与分数／输出，按原atol1e−4／rtol1e−6及RIoU1e−5容差逐帧复现，并精确复现两份完整摘要，通过后才访问TEST。

VAL状态`READOUT_FIXED_ONLINE_VAL_PASS_BEFORE_TEST`，149.9718秒；TEST状态`READOUT_FIXED_TEST_COMPLETE_REVIEW_REQUIRED`，227.4338秒；最终退出0。每帧只1次B特征提取、3次native调用，局部P3／ROI和原图／模型框及sx-sy共用于两个冻结头。VAL／TEST分别887／1440次提取、2661／4320次native调用，头与B前后状态相等，更新0。两个计时含全部检查与三次native审计，不是正式检测FPS。PyTorch allocated峰值360,206,336字节（343.519MiB）、reserved444,596,224字节（424MiB），不是训练零更新小头或总进程显存口径。

TEST数据身份、标注及1440帧（real868＋sim572）逐帧字节与旧封存集合一致；同次B／高斯逐帧输出、决定与完整摘要复现旧报告，**B数值不是临时换基线**。TEST进入前核对匹配的在线VALartifact／源码／权重／runtime，不读其他epoch，不再训练或选参。

本机核对回传包SHA、18份索引内文件SHA及同一91份来源，独立复算VAL／TEST两头的15,348项摘要数值最大差0，逐框重新分解几何最大差8.6927e−6（≤1e−5）。三条件共享精确同次B框／分数／缺输出身份；独立核对旧高斯TEST与均匀选中轮次VAL。未本机重新加载模型、GPU推理或复制权重／ROI缓存。服务器全部原报告／逐帧／progress保留，只回传必要门槛、原预测、摘要、日志和收件记录。

### 自定义指标：同一固定 TEST

| 域 | 条件（VAL已固定权重） | RIoU ↑ | sim A-RMSE(°) ↓ | DFR(%/frame) ↓ | ACI ↑ | MCML_max(帧) ↓ | MCML_mean(帧) ↓ | MRF(帧) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| real | B／检测器24 | 0.8169 | — | 2.5106 | 0.9440 | 4 | 3.5 | 2.29 |
| sim | B／检测器24 | 0.8815 | 1.8822 | 2.4734 | 0.9479 | 0 | 0.0 | — |
| real | 均匀先验／头04 | 0.8159 | — | 2.6757 | 0.9428 | 4 | 3.0 | 2.14 |
| sim | 均匀先验／头04 | 0.8861 | 1.6119 | 2.3051 | 0.9526 | 0 | 0.0 | — |
| real | 高斯先验σ1.5／头03 | 0.8263 | — | 2.7019 | 0.9475 | 4 | 4.0 | 2.43 |
| sim | 高斯先验σ1.5／头03 | 0.8950 | 1.2934 | 2.2299 | 0.9585 | 0 | 0.0 | — |

各组两域TDR_w10均100%、MCML_pass均1；sim MCML_max与MCML_mean均0；sim MRF未报告，保持“—”，不填造值。RIoU为全帧平均，缺输出按0；sim纯周期角RMSE为B1.88217684°、均匀1.61186713°、高斯1.29340460°，本组协议角RMSE与纯角RMSE四舍五入后一致。real不作角度真值精度声明。

### 中心与覆盖（不能混用分母）

三条件real均：输出帧中心命中 **858/859＝99.8836%**；输出覆盖 **859/868＝98.9631%**；全帧中心正确覆盖 **858/868＝98.8479%**。sim三个分母均 **572/572＝100%**。中心命中仅统计输出帧，阈值为原15px；历史终端real/R_center约98.85%使用全帧分母，不能写成输出帧命中率。

### 接受、回退和独立几何分量

均匀头real接受858／回退1／无效候选0，sim接受572／回退0；高斯头real接受859、sim572、回退／无效均0。缺输出9帧是B本身缺输出，不计为回退。均匀回退帧`real_seq03_00113`触发`angle_bound`，整框保留B；没有第二次检测、补输出或得分融合。

| 域 | 条件 | 中心误差均值(px) ↓ | 长边相对误差均值 ↓ | 短边相对误差均值 ↓ |
|---|---|---:|---:|---:|
| real | B／检测器24 | 2.662757 | 0.051917 | 0.065412 |
| sim | B／检测器24 | 1.145166 | 0.040338 | 0.044780 |
| real | 均匀先验／头04 | 2.679957 | 0.059258 | 0.066510 |
| sim | 均匀先验／头04 | 1.309000 | 0.036005 | 0.041133 |
| real | 高斯先验σ1.5／头03 | 2.235548 | 0.050015 | 0.077541 |
| sim | 高斯先验σ1.5／头03 | 1.019973 | 0.036804 | 0.044425 |

独立分量统计输出帧，尺寸使用规范长／短轴，不替代在线raw w-h-angle关联。两头TEST无新增中心失败或恢复，中心计数完全一致；real相对B新增RIoU<0.5帧：均匀1、高斯2，sim均0，平均RIoU提高不代表每帧都提高。

### 对照结论与论文写作边界

**支持的结果。** 固定TEST中，高斯相对均匀先验real／sim RIoU分别高0.0104／0.0089，sim角度RMSE低0.3185°，sim DFR低0.0752个百分点／帧，两域ACI分别高0.0047／0.0059。中心计数、得分与输出覆盖保持，高斯两域平均中心误差更低。均匀头sim相对B亦有改善，但real RIoU、DFR和ACI相对B退化。与VAL证据一起，支持B高斯先验对本任务的定位、方向和部分连续性表现具有增量作用。

**同时报告的代价。** 高斯real DFR2.7019高于均匀2.6757及B2.5106；高斯real MCML_mean4.0／MRF2.43也高于均匀3.0／2.14。高斯real短边误差较均匀／B更大，sim长短边误差较均匀更大；新增低RIoU帧也多1帧。因此不能写成时序、尺寸、每帧精度全面改善。高斯real ACI虽更高，不足以消除DFR和恢复指标的代价。

**决定。** 保留原B＋高斯midpoint σ1.5／epoch03；均匀头作为消融对照，固定TEST并不重新决定版本或开启微调。此表可用于第五节的同集合机制对照，VAL保留为选权与开发依据；不能把VAL与TEST表混写，也不能借TEST挑epoch或新参数。

**限定。** 单种子、TEST已多次暴露且σ1.5工程采用发生在先前TEST回传后；新增固定评价不是全新独立泛化确认，不写显著／稳定／普适或先验必不可少。对照保留中性残差相减，不能单独证明减法项或根因。没有新AP、可靠性、深度或控制评价。本轮不修改论文正文。

可用于论文的精简描述：

> 在相同检测器、细化头容量、训练预算和VAL选权规则下，以均匀先验替换B框高斯先验，并在固定TEST上比较。高斯先验提高两域RIoU、仿真角度精度及ACI，保持中心命中和输出覆盖；真实域DFR和恢复指标仍有代价，部分尺寸误差不占优。这支持空间先验在本任务中改善综合几何表现，但不能解释为所有分量或时序指标全面改善。

### 结果入口与可复核命令

- [固定TEST六行对照](midpoint_readout_fixed_test_evidence_20261012/test/compare.json)
- [完整TEST均匀头摘要](midpoint_readout_fixed_test_evidence_20261012/test/uniform_summary.json)
- [完整TEST高斯头摘要](midpoint_readout_fixed_test_evidence_20261012/test/gaussian_summary.json)
- [TEST完成与同次B／旧结果重放依据](midpoint_readout_fixed_test_evidence_20261012/test/completion.json)
- [双头在线VAL门槛](midpoint_readout_fixed_test_evidence_20261012/val/completion.json)
- [本机独立复算与收件](midpoint_readout_fixed_test_evidence_20261012/local_review_receipt.json)

服务器原目录：`work_dirs/port_geometry_midpoint_readout_test_v1/20261012_gpu2_v1/`。下面命令已按顺序完成；已有输出不得覆盖，再核验时使用新后缀，通常无需重复运行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_port_geometry_midpoint_readout_test_v1.py
CUDA_VISIBLE_DEVICES="" python crane_project/tools/eval_port_geometry_midpoint_readout_test_v1.py   --stage check --out-dir work_dirs/port_geometry_midpoint_readout_test_v1/20261012_gpu2_v1/static
CUDA_VISIBLE_DEVICES=2 python crane_project/tools/eval_port_geometry_midpoint_readout_test_v1.py   --stage val --gpu 0 --out-dir work_dirs/port_geometry_midpoint_readout_test_v1/20261012_gpu2_v1/val
CUDA_VISIBLE_DEVICES=2 python crane_project/tools/eval_port_geometry_midpoint_readout_test_v1.py   --stage test --gpu 0   --val-report work_dirs/port_geometry_midpoint_readout_test_v1/20261012_gpu2_v1/val/completion.json   --out-dir work_dirs/port_geometry_midpoint_readout_test_v1/20261012_gpu2_v1/test
```

回传包2,529,469字节、SHA256 `88b00bf5f44fd050e70c3e34002fc35a8814ed004162616edf41e316fa061b79`，包含28份必要文本报告／预测；没有源码、权重或缓存副本。本机与服务器本次传输压缩包已删除且核验不存在，清理状态见本机receipt，服务器原始输出保留。

## 论文用途证据精简（2026-10-12）

按用户要求核对四个证据目录后，18份重复日志／退出码／CSV／静态与旧环境失败副本移入系统废纸篓，恢复目录`/Users/mac/.Trash/symEOOD_paper_evidence_redundant_20261012_103648`，清单manifest.json保留原路径、大小和SHA。本机未清理服务器，未改原始数值／权重／选权／源码或重新运行实验。

论文读取入口和本页TEST／VAL表保持；AP原生报告／CPU PR及TEST源输入保留，正式完整24轮选择和选中逐帧结果保留，在线VAL门槛及TEST两头完整预测／摘要／来源保留。原artifact／历史核验清单索引的是精简前完整运行，不能声称本地现有子集就是全量服务器输出；当前保留内容及恢复记录见各目录README和local_review_receipt。

## 主证据目录合并（2026-10-12）

当前只保留`midpoint_readout_fixed_test_evidence_20261012/`作为正式机制对照的主目录，内含`test/`论文主结果、`val/`在线评价门槛及`train/`训练完成／24轮选权依据。原formal目录9份必要文件逐字节迁入train，重复README移入废纸篓，原目录移除；本页链接同步更新。没有删除权重选择依据，也没有把VAL或训练成绩改写为TEST。原报告、原artifact、原source快照及实验数值保持；原服务器目录不变。

仅更新本机测试夹具读取证据的路径，评价源码／协议及训练87份来源不改。当前评价来源清单相应绑定新的夹具SHA；已完成VAL／TEST报告内保存的当时91份来源不重写、不冒充新来源下重跑。本轮是文件整理，不重新训练／推理／选权；未来需要重新运行评价时，仍需与当次来源匹配的在线VAL门槛，不复用不匹配的旧门槛。
