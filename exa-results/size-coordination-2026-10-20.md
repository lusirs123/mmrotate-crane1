# A20后双边协调：定向文献筛选（2026-10-20）

## 范围与结论

本轮使用Exa按联合几何、已有收益保留、优化竞争三个角度检索；15次成功检索请求的numResults合计74条，含同文不同入口与重复结果，不等于74篇核读论文。另核读原文与作者代码；Exa正文截断时用网页读取工具补读已知原文／raw源码。没有克隆、安装或运行作者模型，没有改项目代码、核心文档、权重、数据或实验合同。网页代码按本次访问的main/master读取，未固定commit，不作为可复现实验源码包。

结论：优先审查“联合几何目标下，固定A20的有效纠错如何保留”的训练设计；教师误差约束提供保护思路，联合分布提供关联表示备选。没有找到能直接保证准确边不损伤、跨域收益或深度改善的可移植模块。本轮未形成训练候选，更未证明新颖性。

本地依据：[核心文档第14节](../docs/detection/尺寸精度主要结果与对比分析_20261006.md#third-contribution-writing)、[历史过程记录](../docs/detection/尺寸优化专用记录_20261016.md)、[上轮定向文献](../docs/detection/S037后定向文献与机制选择_20261019.md)。A048已试比例监督，S021已试单边加权；S013—S015已检验分布读出／监督的特定适配；S035未证明系统性梯度冲突；S038／S039的Q对齐没有超过共同控制。上述事实限制本轮候选解释。

## 1. 优先核对的保护思路：教师误差约束

**Learning Efficient Object Detection Models with Knowledge Distillation，NeurIPS 2017。**

[会议原文§3.3、式4](https://proceedings.neurips.cc/paper/2017/file/e1e32e235eee1f970470a3a6658dfdd5-Paper.pdf)。原文比较学生和教师相对GT的回归误差；学生未优于教师达到指定margin时，施加额外GT回归惩罚。并非让学生始终贴近教师坐标。本轮未核实作者官方可用实现，不提供第三方同名仓库冒充代码。

适配推断：固定A20可作为训练时的误差参照，而非不可超越的预测目标。这样有机会在修正比例时明确检查已有效修正的边是否变差；训练GT可用，推理不能使用GT来选择M／A20／新模型。

限制：原式是整框回归误差，不能防止一条边改善掩盖另一条边损伤；它是软惩罚，不是逐样本或VAL硬保证。若改为逐边误差参照，这是本项目的新适配，不能冒称论文已验证。它还可能实质上只是样本动态加权，必须与S021及同预算GT回归／普通比例监督比较，不能仅加一个loss就认定新机制。A20未必比M的near边好，不能仅守住A20而遗漏M保护。

## 2. 联合表示备选：多变量误差分布

**Deep Mixture Density Network for Probabilistic Object Detection，按arXiv修订版核读。**

[原文](https://arxiv.org/pdf/1911.10614)、[版本页](https://arxiv.org/abs/1911.10614)。原文学习框坐标的联合分布，用Cholesky参数化精度矩阵以表达边之间关联，并讨论均值及混合分量推理解码；协方差无需参与最终点预测。未核实可直接使用的作者官方源码。旧标题的OpenReview撤回投稿不能当成ICLR录用，本文不据该入口标会议身份。

适配推断：可研究两条规范log边长残差的二维联合表示，与独立分布比较。它不同于旧方案仅分别预测两边桶分布；但仅由独立坐标改为a/r坐标，也不自动等价于学习了样本相关的协方差。

限制：概率拟合改善不等于两边MAE／比例改善。增大方差可能降低困难残差的惩罚，应核查均值预测是否真正改准。有限且存在固定比例生成标签的TRAIN，可能只学到视频比例规律。暂不扩为多混合分量或与保护loss一起叠加；需要先说明非对角关联是否有可检验价值。该法没有新增视觉信息或准确边保护保证。

## 3. 定位知识保留参考：LD／Rotated-LD

**Localization Distillation for Dense Object Detection，CVPR 2022；扩展版TPAMI 2023。**

[会议原文](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)、[作者仓库](https://github.com/HikariTJU/LD)、[本轮实读ld_head.py](https://github.com/HikariTJU/LD/blob/main/mmdet/models/dense_heads/ld_head.py)。作者仓库提供Rotated-LD入口，本轮未核读其旋转实现全链路。

代码的loss_single将学生和教师四边分布logits展开，调用KL蒸馏，并对额外定位区域单独计算损失；这不同于仅惩罚最终坐标差。可借鉴训练期保留定位知识的接口，不能把valuable localization region理解成保证教师在该处正确。

A20输出的是点尺寸／局部向量，不是已校准的长短边概率分布；不能把空间81个位置直接冒称尺寸概率桶。接入LD需要明确教师分布含义，可能引入新头与额外容量。全面模仿A20也会保留其比例错误，故不优先直接移植。

## 4. 新近表示参考：Polar Loss

**Polar loss: A novel IoU-free loss function for bounding box regression in FCOS for autonomous driving systems，International Journal of Cognitive Computing in Engineering，2026。**

[出版页及摘要](https://doi.org/10.1016/j.ijcce.2026.05.005)。摘要将对角点转为极坐标，分别表达尺度与形状。本轮仅核到出版摘要，未核到完整公式及作者官方实现，不能据此制定数值训练合同；同名零样本检测Polar／Polarity Loss仓库不是这篇论文。

可借鉴尺度与形状分开表达的思路；极坐标角不是OBB朝向。适配时必须使用规范尺寸，避免沿M轴的GT投影范围再次混入中心／角度误差。项目已有a/r与比例监督，因此仅更换参数化并非充分新意；也没有证据表明它保护near或解决Real迁移。

## 5. 历史文献补核：D-FINE

**D-FINE，ICLR 2025。**

[会议原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html)、[作者criterion](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_criterion.py)。本轮实读loss_ddf：使用detach后的teacher_corners、温度KL及权重，进行层间定位分布蒸馏。项目S013借鉴过分布监督，但没有完整实现这种层间自蒸馏，不能将其称为已被项目全面否定，也不能说是本轮新发现。

A20缺少DETR多层分布输出；移植会同时改变结构、表达和训练路径，首轮对照难以单独归因。四条边的分布细化不等于显式学习长短边联合协方差。不建议为了套用D-FINE重建整套多阶段头。

## 6. 条件备选：CAGrad

**Conflict-Averse Gradient Descent for Multi-task Learning，NeurIPS 2021。**

[原文](https://arxiv.org/html/2110.14048)、[作者仓库](https://github.com/Cranial-XIX/CAGrad)、[实读toy.py::cagrad](https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py)。代码根据任务梯度内积求组合系数；方法关注平均目标与较差局部改进之间的平衡。

它可作为多目标更新对照，但任务批平均梯度不保证每帧每条near边受保护，也不提供修正方向的新视觉依据。S035只支持局部干扰，未支持系统性冲突为主要根因。因此CAGrad／PCGrad不列首选，更不同时叠加在教师约束和联合分布上。

## 7. 下一步建议：先限定一个问题，不堆论文模块

优先审查：**保持A20读出和推理接口时，能否定义允许继续纠错、同时检查有效收益损失的逐边训练约束，并与联合几何目标相容？** 以2017教师误差约束作思想参照，以LD作定位知识保留的实现边界参照，不直接复制两篇的完整蒸馏框架。

审查交付应回答：

1. 用已有完整TRAIN／VAL结果确认A20中哪些逐帧双边取舍具有共同支持；GT只参与训练目标或离线分析，不成为在线门控。已有统计直接复用，补缺项前另行明确范围。
2. 是否保留M near保护与A20有效纠错两种不同职责？能否避免将A20原有错误冻结为正确答案？
3. 候选与旧比例loss、S021动态贡献加权的数学和更新差异是什么？若只是同类加权，应明确这是控制／复核，而非新贡献。
4. 用相同A20起点、优化器初始化和预算的续训控制区分额外训练收益；比例监督本身的效应也需可分辨。若对照无法收束，暂不形成训练合同。

联合协方差表示保留为独立备选，不与首选同时叠加。原生P3、多尺度、Q对齐、额外时序模块不自动追加。现有文献没有解决“推理时如何稳定知道某条边该放大还是缩小”的证据缺口；训练保护方案只能成为待检验假设。

论文仍按两张核心表报告，主张不扩大为已解决跨域或深度。开发与选权仅依据合法TRAIN／VAL安排；TEST和Unknown-02只作冻结描述，不用于挑损失、补偿或权重。当前正式M保持，第三个创新点仍待新机制与新结果支持。
