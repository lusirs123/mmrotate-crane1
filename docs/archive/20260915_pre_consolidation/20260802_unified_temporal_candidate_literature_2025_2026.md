# 统一跨尺度因果时序候选选择：2025–2026 文献映射

日期：2026-08-02

## 1. 当前问题与检索边界

本报告服务于以下已预注册方向：

```text
native S14 + frozen S7 candidate pool
  -> candidate evidence
  -> causal temporal association/reranking
  -> native-safe top-1
  -> post-selection causal OBB stabilization
```

目标不是按 `seq02_dark`、`seq03_small` 或亮度/尺度片段人工路由，而是对所有未知连续
视频使用同一个候选选择器。检索聚焦 2025–2026 年的 oriented localization quality、
small-object association、DINO temporal representation、continuous video association、
pseudo-label filtering 和 frozen-backbone dense readout。只收录已由 CVF、AAAI 或 arXiv
原始页面核实的英文论文。

已有负结果必须继续生效：BrightAug 的 RegQuality/PQA v1–v3、QFL、去 teacher-force
手工时序重锚均已封存。新文献只能用于当前 DINO S14+S7 top-100 候选池；不能把旧
PQA 换名重跑，也不能在 target 切片上扫描手工时序权重。

## 2. 第一优先级：直接支撑当前统一方案

### TQ-01 Pixel-level Quality Assessment for Oriented Object Detection

- Zhu, Yunhui; Huang, Buliao. AAAI 2026, 40(16): 14005–14013.
- DOI: `10.1609/aaai.v40i16.38411`
- 原始页面：https://ojs.aaai.org/index.php/AAAI/article/view/38411
- 核心证据：旋转检测最终依赖候选排序，而分类分数不必然反映定位质量；论文用像素级
  空间一致性替代结构耦合的 box-level IoU prediction。
- 可借鉴：把“candidate quality 必须与真实 OBB 定位质量对齐”作为方法动机和相关工作。
- 不可直接照搬：项目的 BrightAug PQA v1–v3 已正式失败；该论文不能授权 PQA v4。
  若当前 S7 ROI 池使用质量信息，必须是新候选空间上的独立预检，并优先采用排序/关联
  证据，而不是再次使用 quality-only top-1。

### TT-01 Dist-Tracker: A Small Object-aware Detector and Tracker for UAV Tracking

- Wang et al. CVPR Workshops 2025, pp. 6667–6675.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.html
- 核心证据：小目标的轻微位置抖动会使 IoU 关联不稳定；其 FLIT tracker 联合 L2 和 IoU，
  用中心距离补偿小目标的位置不确定性。
- 直接映射：`seq03_small` 的关联不能只看 rotated IoU，应至少联合归一化中心距离；
  `seq02_dark` 大目标仍可让 IoU 保持较大权重。
- 风险：场景是红外多 UAV，且为 workshop/challenge 工作；适合作为关联代价设计依据，
  不应直接宣称其检测收益可迁移到本项目。

### TT-02 Exploring Temporally-Aware Features for Point Tracking (Chrono)

- Kim et al. CVPR 2025, pp. 1962–1972.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Exploring_Temporally-Aware_Features_for_Point_Tracking_CVPR_2025_paper.html
- 核心证据：Chrono 在 DINOv2 上加入 temporal adapter，使冻结/预训练视觉表征获得长期
  时序感知，并能以简单特征匹配完成高质量点跟踪。
- 直接映射：复用 DINO ROI/patch embedding 做跨帧外观一致性，而不是另训一个依赖
  target 片段身份的路由器。
- 风险：原任务是 point tracking，不是旋转框检测；优先借鉴 temporal feature adapter 和
  matching 表征，不照搬输出头。

### TT-03 COVTrack: Continuous Open-Vocabulary Tracking via Adaptive Multi-Cue Fusion

- Qian et al. ICCV 2025, pp. 10054–10063.
- 原始页面：https://openaccess.thecvf.com/content/ICCV2025/html/Qian_COVTrack_Continuous_Open-Vocabulary_Tracking_via_Adaptive_Multi-Cue_Fusion_ICCV_2025_paper.html
- 核心证据：连续标注轨迹对稳健 tracker 学习重要；方法按帧内和帧间置信动态融合
  appearance、motion 和 semantic cues。
- 直接映射：支持统一使用分类/质量、中心运动、旋转几何和 DINO 外观多线索，而不是
  固定一套 IoU 权重或按暗光/小目标切换。
- 风险：开放词汇多目标跟踪比本项目复杂。当前单类/近单目标场景应提炼为轻量因果
  scorer，不引入完整 OVMOT 框架。

### TT-04 VOVTrack: Exploring the Potentiality in Raw Videos for Open-Vocabulary Multi-Object Tracking

- Qian et al. ICCV 2025, pp. 7472–7482.
- 原始页面：https://openaccess.thecvf.com/content/ICCV2025/html/Qian_VOVTrack_Exploring_the_Potentiality_in_Raw_Videos_for_Open-Vocabulary_Multi-Object_ICCV_2025_paper.html
- 核心证据：利用无标注原始视频构造 self-supervised object similarity learning，服务
  跨帧关联；同时将 tracking-related object states 用于动态目标定位和分类。
- 直接映射：若存在严格隔离的无标注 target-train 视频，可学习跨帧 DINO ROI similarity，
  不需要 target GT；这比预定义片段路由更具迁移性。
- 风险：必须先证明 target-train 与 dev/test 隔离；否则只允许在 source 连续序列训练。

## 3. 第二优先级：候选排序、读出与伪标签流程

### TQ-02 RARE: Learn to RAnk and REtrieve for Monocular 3D Object Detection

- Park et al. CVPR 2026, pp. 11556–11566.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2026/html/Park_RARE_Learn_to_RAnk_and_REtrieve_for_Monocular_3D_Object_CVPR_2026_paper.html
- 核心证据：把 confidence estimation 作为相对质量排序问题，而不是只拟合绝对质量；
  联合 point-wise quality alignment 与 pair-wise ordering，再从多候选中 retrieve 最优解。
- 直接映射：当前 `64/64` 候选池更需要 relative rank-and-retrieve，而不是全局 affine 或
  frame-wide delta。
- 风险：原任务是单目 3D DETR。其价值在损失和候选检索思想，不是网络结构。

### TR-01 LiDeRe: A Lightweight Readout for Fast and Data-Efficient Dense Prediction

- Lüddecke et al. CVPR 2026, pp. 2959–2971.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2026/html/Luddecke_LiDeRe_A_Lightweight_Readout_for_Fast_and_Data-Efficient_Dense_Prediction_CVPR_2026_paper.html
- 核心证据：冻结大型视觉 backbone 上的轻量 interpolation+attention dense readout 能以
  较低参数/数据成本恢复细粒度预测，并覆盖 object detection。
- 直接映射：继续支撑 S7 作为冻结 DINO 的高分辨率候选生成器。
- 当前裁决：S7 已达到 `seq03_small` R@100/geometry `64/64`，因此 LiDeRe 升级不是
  当前优先级；先解决候选选择，只有新数据再次证明 candidate coverage 不足才升级读出。

### TR-02 Real-Time Object Detection Meets DINOv3 (DEIMv2)

- Huang et al. arXiv 2025, arXiv:2509.20787.
- 原始页面：https://arxiv.org/abs/2509.20787
- 核心证据：Spatial Tuning Adapter 将 DINOv3 单尺度输出转为多尺度特征，以细节补充
  强语义，并兼顾实时检测成本。
- 直接映射：支持“冻结/预训练语义 + 轻量空间适配”的结构合理性。
- 当前裁决：不应替换现有 DINOv2/S7；作为未来压缩或正式 LiDeRe 升级后的结构参考。

### TP-01 Leveraging Temporal Cues for Semi-Supervised Multi-View 3D Object Detection

- Park et al. CVPR 2025, pp. 27401–27412.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2025/html/Park_Leveraging_Temporal_Cues_for_Semi-Supervised_Multi-View_3D_Object_Detection_CVPR_2025_paper.html
- 核心证据：单纯置信阈值不足以保证伪标签定位质量；论文使用前向/反向预测集成、
  tracking infill 和辅助检测头过滤改善时序伪标签。
- 直接映射：正式 target-train 阶段不应按单帧 S7 分数生成伪标签；应要求跨帧持续、
  native/S7 一致或辅助候选质量证据。
- 风险：原任务为多视角 3D；只借鉴伪标签过滤原则。

### TP-02 Partial Weakly-Supervised Oriented Object Detection (PWOOD)

- Liu et al. CVPR 2026, pp. 27644–27654.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Partial_Weakly-Supervised_Oriented_Object_Detection_CVPR_2026_paper.html
- 核心证据：Orientation-and-Scale-aware Student 与 class-agnostic pseudo-label filtering
  用于降低旋转检测对固定伪标签阈值的敏感性。
- 直接映射：正式学生训练可避免单一固定 score threshold，并显式保护 orientation/scale。
- 风险：PWOOD 使用部分弱标注，不等同于本项目纯无标签 target-train；需改写监督来源。

## 4. 第三优先级：可选实现参考

### TT-05 Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better

- Lai; Vedaldi. CVPR 2025, pp. 22809–22819.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2025/html/Lai_Tracktention_Leveraging_Point_Tracking_to_Attend_Videos_Faster_and_Better_CVPR_2025_paper.html
- 证据：显式 point tracks 可作为轻量插件改善视频模型的时序对齐和长程一致性。
- 用法：若短期 box association 仍不稳，可用内部 DINO point tracks 补充 motion cue。
- 风险：额外 tracker 的误差会传播；不应作为第一版依赖。

### TT-06 Multiple Object Tracking as ID Prediction (MOTIP)

- Gao; Qi; Wang. CVPR 2025, pp. 27883–27893.
- 原始页面：https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html
- 证据：以历史 trajectory object features 为上下文直接预测当前检测 ID，可替代复杂手工
  cost matrix，并从训练数据学习关联。
- 用法：若轻量多线索 scorer 的 source 上界不足，再考虑小型 trajectory decoder。
- 风险：当前单目标任务不需要完整 ID dictionary/多目标系统，第一版不应采用。

## 5. 证据—论点映射


| Source ID | Abstract-level finding                                   | Supported claim                            | Citation slot                      | Risk                                   |
| ----------- | ---------------------------------------------------------- | -------------------------------------------- | ------------------------------------ | ---------------------------------------- |
| TQ-01     | OBB 分类分数与定位质量可能错位，像素级一致性可估计质量   | 当前正确 S7 候选不能只按分类分数选 top-1   | Related Work：OBB quality ranking  | PQA v1–v3 已失败，只用于动机/新池预检 |
| TQ-02     | 相对 ranking+retrieval 可改善候选质量顺序                | 应学习候选间相对顺序而非 frame-wide delta  | Method：candidate evidence         | 原任务是 3D DETR                       |
| TT-01     | 小目标 IoU 因位置抖动失稳，L2+IoU 更稳                   | 小目标时序关联必须加入中心距离             | Method：association cost           | workshop/红外 UAV                      |
| TT-02     | DINOv2+temporal adapter 可形成时序感知特征               | DINO ROI embedding 可承担跨帧外观 cue      | Method：appearance cue             | point tracking 非 OBB                  |
| TT-03     | 连续视频需自适应融合 motion/semantic/appearance          | 暗光与小目标可用统一 multi-cue selector    | Method：unified causal selector    | OVMOT 结构过重                         |
| TT-04     | 无标注原始视频可自监督学习 object similarity             | 隔离 target-train 可用于无 GT 时序关联学习 | Training：target-train consistency | 数据隔离是硬门                         |
| TR-01     | 冻结 backbone 的轻量 readout 可恢复 dense detail         | S7 候选生成方向有文献依据                  | Related Work：frozen dense readout | 当前 coverage 已足够                   |
| TP-01     | tracking infill/多证据过滤改善时序伪标签                 | 正式伪标签不能仅用单帧 S7 confidence       | Training：pseudo-label filtering   | 3D 到 2D 迁移                          |
| TP-02     | OBB student 与 class-agnostic filtering 降低固定阈值敏感 | 学生训练应保护方向/尺度并减少静态阈值依赖  | Training：oriented student         | 弱标注设定不同                         |

## 6. 综合裁决与实施优先级

文献不支持再次训练 lane-wide quality suppression，也不支持把旧 PQA 换名重跑。最小且
与历史负结果区分清楚的下一步是：

```text
P0：固定 native+S7 候选，不训练检测器
P1：在 source 连续序列训练/拟合轻量 multi-cue causal association
    cues = calibrated score + normalized center distance + rotated IoU
           + log-scale/periodic-angle consistency + DINO ROI similarity
P2：native-safe override、reset 和 exact fallback
P3：source exact-retention + source temporal MCML/DFR/ACI gate
P4：全部冻结后只做一次固定 target-dev diagnosis
```

第一版应以 Dist-Tracker 的 `L2+IoU` 和 COVTrack 的 multi-cue fusion 为核心，Chrono
只提供 DINO temporal feature 依据。PQA/RARE 只在现有 score+association 仍无法从
top-100 候选中选出稳定轨迹，且 source 候选排序预检显示明确监督支持时，才授权独立的
candidate-level relative ranking；不得使用 quality-only top-1。

最终方法的新颖性不应声称来自单个模块，而应定位为：在冻结 DINO 旋转检测器中，将
高分辨率补充候选、source-safe 多线索因果关联、native fallback 和后置周期 OBB 稳定
统一起来，以同时处理暗光语义排序和小目标空间分辨率问题，并取消 target-scope 路由。
