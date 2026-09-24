# DINO → SymEOOD 初步蒸馏试验与对话迁移记录（2026-09-23）

## 当前结论与任务定位

当前阶段是轻量学生蒸馏可行性验证。目标是在训练时利用冻结 DINOv2 教师，部署时只运行 SymEOOD 学生。先前运行时调用 DINO 的 scoped 融合、DINO 独立检测诊断，与本轮缓存特征蒸馏是不同实验身份。

本轮 warmstart V2 相对于同预算无蒸馏对照，只增加 1 帧中心命中，real 最长连续缺测均为 62 帧；未获得实质连续检测收益。结论仅适用于这套前景特征损失、初始化和训练预算，不等于所有蒸馏方法无效，也没有多随机种子统计显著性结论。

## 试验目的、设计与针对困难

历史关注 real_seq02 远距片段 [2,41]、暗光片段 [137,169] 与 real_seq03 小目标片段 [129,192]。这些是已暴露 TEST 的诊断标签，不允许据此挑训练样本或调整参数。seq03 抓料可能影响形状，但不从正式 TEST 删除，也不把动作影响当作已证实因果。

从头训练特征蒸馏 V1 曾丢失 K1 输出。V2 回答：从已选中 K1 初始化，DINO 特征损失是否比普通继续训练有额外收益？两组均从 K1 epoch_20 初始化，计划训练4 epoch，SGD lr=0.00025、momentum=0.9、weight_decay=0.0001，warmup100步、ratio0.1、step=[3]，梯度裁剪10。source train2033帧 + train_sim748帧；VAL738帧；TEST992帧。计划每卡batch2、两卡顺序训练；这些训练设置来自仓库配置，包内没有训练日志，未据此验证实际启动参数。

两组学生均有零初始化分类残差适配器；control关闭蒸馏损失。distill读取1024通道DINOv2 ViT-L/14缓存，使用单层FPN（feature_level=0）、前景余弦对齐、权重0.05、最多4096 tokens。protect_geometry=True 对蒸馏输入执行detach，蒸馏梯度直接更新适配器和训练期投影，不直接更新主干/FPN；检测损失仍更新检测网络，因此不能声称几何完全受保护。尚未蒸馏抓斗教师RPN/ROI的任务输出。

源域VAL从epoch1–4按既有自定义规则选权，两组均选epoch_1。原规则包含加权中心召回约束、MCML≤5及TDR、中心、ACI、角度软评分；本次未依据TEST重选。当前metric_v3只是输出目录名，报告metric_protocol_version仍为2。

## 正式 TEST 表（直接读取归档JSON）

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

## 配对结果（完整记录见 paired_test_audit.json）

| 分组 | 对照输出 | 学生输出 | 对照中心命中 | 学生中心命中 | 共同输出RIoU：对照→学生 |
|---|---:|---:|---:|---:|---|
| real | 309 | 310 | 304 | 305 | 0.689634719 → 0.689516215 |
| real/seq02 | 121 | 121 | 117 | 117 | 0.685973641 → 0.685302251 |
| real/seq03 | 188 | 189 | 187 | 188 | 0.691991052 → 0.692228394 |
| sim | 572 | 572 | 572 | 572 | 0.828363051 → 0.828201647 |

学生独有输出记录：`[{"frame_key": "real_seq03_00186", "domain": "real", "sequence": "seq03", "frame": 186, "gt_short_edge_px": 22.40755271911621, "baseline": {"output": false, "center_hit": false, "riou_hit": false, "center_error_px": null, "riou": null}, "student": {"output": true, "center_hit": true, "riou_hit": false, "center_error_px": 6.655543107343824, "riou": 0.49407209503787036}}]`。

real共有309帧双方输出，学生共同帧RIoU略降；seq02双方均缺99帧，输出集合完全相同；seq03仅补回1帧。real的RIoU命中总数均为274，不能将新增中心命中直接称为新增完整框命中。sim均完整输出，几何差异微小。

## 必须纠正的历史说明

1. 旧选权SHA 551a255db55e0125141d35239e207298868326e5909895470dcefb527bf2f4ac 与本地统一学生推理配置 crane_symeood_k1_dino_semantic_student_v1.py 完全一致。此前把它解释为“选权后配置被修改”，证据不足；更符合现有证据的是VAL使用学生推理配置，而提供的TEST命令错误换成训练配置。原设计文档也要求用统一学生配置选权。哈希保护本身正常，但之前给出的命令和原因解释有误，不应为消除保护而重写旧哈希。

2. 用户粘贴过real R_center=96.19%、RIoU=0.7937、MCML=6的历史表。本包未定位到这一整组报告的可靠身份，之前直接确认其为独立native-S14 DINO正式TEST是不成立的。包中formal_integrated_nms05与formal_integrated_test_refactor绑定scoped配置及BrightAug epoch20，包含融合结果：real条件中心97.39%、另列全帧25px中心72.38%、MCML39。它们不能当作同口径独立教师上限。需要对应96.19/6的原始报告、配置、权重身份及预测，才能重算同协议教师比较。

3. 实际配对文件名是 work_dirs/crane_symeood_k1_dino_warmstart_v2_paired_test_audit.json（及.md），此前漏写crane_symeood前缀导致打包失败。

## 当前改进方向与停止条件

本轮停止继续围绕这一个TEST净增帧调节余弦损失、阈值或训练轮数。保留当前负结果及同预算对照。下一步候选方向是任务相关的选择性蒸馏，但尚未证明有效，也未在本轮实现。

先确定实际教师身份和源域教师正确、学生不足的监督支持，复用已有source质量与覆盖诊断；不要重复把源域缺少真实小目标验证支持包装成新发现，不重新划分数据。存在可信监督时，以K1初始化，保留GT检测损失，优先迁移对象与邻近背景的区分或可靠类别响应；异构检测头需明确区域/候选对应，不直接复制权重或全面照搬教师框。

可考虑源域清晰教师视图与学生合理光照扰动的对应，以及有限后段学生参数接受蒸馏梯度。加入原有能力保留约束是待验证选择，不保证覆盖率保留；全部新增机制不能一次堆叠。教师几何监督需经source GT核验，小目标分辨率/尺度问题独立讨论。新方案先固定设计、同预算对照、source VAL选权，再冻结评价；已暴露TEST只能诊断，不能声称未见测试的无偏确认。

资源目标：训练优先缓存教师输出，不加载在线大教师计算图；控制缓存与GPU张量上界，顺序运行实验。最终单学生推理必须测延迟和峰值显存，不能凭参数少宣称加速；本包没有新延迟测量。

## 证据位置、验证与复现

本地仓库 /Users/mac/Documents/paper/symEOOD；服务器 /media/omnisky/personal_files/ljj/symEOOD；Python3.8 mmrotljj。原始压缩包 /Users/mac/Downloads/k1_dino_warmstart_test_analysis_20260923.tar.gz。SHA256为47a9c479d8f6e1ed8ff81013e945071f833ac315d56f9a99f4ae27c57dfdabe9。全部包内成员路径与哈希保存在相邻evidence/dino_warmstart_v2_20260923/archive_manifest.json。

关键原文已复制到 [证据目录](evidence/dino_warmstart_v2_20260923/)，不会依赖/tmp解压目录。control_selection.json、distill_selection.json保存选权；control_test.json、distill_test.json保存正式指标；paired_test_audit.json保存992帧；两份formal摘要只作为历史身份核对证据。

本轮实际验证：压缩包SHA与服务器记录一致；两组选权checkpoint哈希与TEST及配对审计一致；config哈希三方一致；配对审计绑定的TEST文件SHA与实际文件一致；配对记录992帧。未重跑训练、模型推理或GPU测试；缺少PKL/Task1_grab与权重，不能独立重算框几何或验证实际训练参数。

复现入口：ckpt_sweep.py --final-test-from 必须使用选权时同一config并保持config哈希，sweep目录必须与JSON所在目录一致。当前两组位于 work_dirs/crane_symeood_k1_dino_warmstart_{control,distill}_v2/ckpt_sweep_metric_v3/final_test/epoch_1/；配对工具 audit_k1_dino_student_paired_test_v1.py 使用 --gt-dir、两组 --*-report/--*-pkl/--*-pred-dir 读取已存在预测即可，不需要重跑模型。

## 新对话接续提示

先读取本文与证据JSON。用户当前目标是DINO能力迁移和学生轻量部署，暂不写论文正文。不要混淆DINOv2特征教师、抓斗DINO检测器、scoped运行时融合、Base V3观测系统和本轮warmstart学生。V2已经完成初步对照且没有实质收益；后续讨论任务相关蒸馏时先检查是否与历史失败方法重复。优先核实教师身份与监督支持，避免反复让用户重新选权、打包或重复完整诊断。
