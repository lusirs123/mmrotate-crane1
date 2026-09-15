# Unified pairwise takeover ranker V2 evidence map

| Source ID | Citation | Source type | Abstract-level finding | Usable fact | Supported claim | Citation slot | Risk |
|---|---|---|---|---|---|---|---|
| E1 | Jiang et al., *Acquisition of Localization Confidence for Accurate Object Detection*, ECCV 2018, https://openaccess.thecvf.com/content_ECCV_2018/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.html | Peer-reviewed conference paper | IoU-Net predicts box-to-GT IoU as localization confidence and uses it to preserve accurately localized boxes. | Localization quality can be learned separately from classification confidence. | Supervise candidate quality with source GT overlap instead of relying only on detector class score. | V2 motivation | Direct |
| E2 | Tan et al., *Learning to Rank Proposals for Object Detection*, ICCV 2019, https://openaccess.thecvf.com/content_ICCV_2019/html/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.html | Peer-reviewed conference paper | Learns IoU-based proposal ranking scores and uses level-based hard-pair sampling. | IoU-guided pairwise ranking and hard-pair mining are established detection mechanisms. | Rank S7 candidates by relative localization quality and mine dangerous competitors. | V2 ranking loss | Direct |
| E3 | Liu et al., *RankDetNet: Delving Into Ranking Constraints for Object Detection*, CVPR 2021, https://openaccess.thecvf.com/content/CVPR2021/html/Liu_RankDetNet_Delving_Into_Ranking_Constraints_for_Object_Detection_CVPR_2021_paper.html | Peer-reviewed conference paper | Aligns confidence-score order with IoU order through IoU-guided ranking constraints. | Pairwise confidence/IoU ordering can reduce classification-localization mismatch. | Train the takeover order against source-GT quality differences. | V2 ranking loss | Direct |
| E4 | Oksuz et al., *Rank & Sort Loss for Object Detection and Instance Segmentation*, ICCV 2021, https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html | Peer-reviewed conference paper | Ranks positives above negatives and sorts positives by localization quality. | Sorting usable candidates by localization quality is a valid detection objective. | Preserve ordering among multiple usable S7 candidates, not only binary foreground/background separation. | V2 listwise loss | Direct |
| E5 | Choi et al., *Gaussian YOLOv3*, ICCV 2019, https://openaccess.thecvf.com/content_ICCV_2019/html/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.html | Peer-reviewed conference paper | Models localization uncertainty and uses predicted uncertainty to reduce false positives. | Detection decisions can use learned localization uncertainty rather than point confidence alone. | Add uncertainty-aware abstention to S7 takeover. | V2 uncertainty | Indirect: uncertainty target and detector differ |
| E6 | Geifman and El-Yaniv, *SelectiveNet*, ICML 2019, https://proceedings.mlr.press/v97/geifman19a.html | Peer-reviewed conference paper | Jointly optimizes prediction and rejection and improves risk-coverage trade-offs. | A learned reject option is preferable to unconditional prediction under uncertain evidence. | Fall back to native when takeover evidence is uncertain. | V2 abstention | Indirect: generic selective prediction |
| E7 | Danish et al., *Improving Single Domain-Generalized Object Detection: A Focus on Diversification and Alignment*, CVPR 2024, https://openaccess.thecvf.com/content/CVPR2024/html/Danish_Improving_Single_Domain-Generalized_Object_Detection_A_Focus_on_Diversification_and_CVPR_2024_paper.html | Peer-reviewed conference paper | Source diversification plus cross-view classification/localization alignment improves single-domain generalized detection. | Augmentation should be coupled with multi-view alignment rather than used only as independent replacement views. | Add clean/aug ranking consistency using source-only views. | V2 consistency | Direct at strategy level |
| E8 | Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*, ICLR 2020, https://openreview.net/pdf?id=ryxGuJrFvS | Peer-reviewed conference paper | Group DRO minimizes worst-group loss; regularization and early stopping are important for worst-group generalization. | Average gains can hide weak-group failure, and worst-group optimization is a recognized remedy. | Balance real/sim source groups and select by worst-group behavior. | V2 group robustness | Indirect: not object-detection-specific |

## Boundary

No cited paper directly establishes the complete proposed rule
`delta-RIoU prediction + uncertainty lower confidence bound + native fallback`
for a frozen DINOv2 native/S7 rotated detector. That rule is a project-specific
synthesis of E1--E8 and remains a hypothesis until source-only and unseen-sequence
experiments validate it.

# OBB 倾斜补偿单目深度公式证据映射

| Source ID | Citation | Source type | 可用事实 | 本文支持的论断 | 证据关系 | 风险边界 |
|---|---|---|---|---|---|---|
| D1 | Ku, Pon, and Waslander, *Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction*, CVPR 2019, https://openaccess.thecvf.com/content_CVPR_2019/papers/Ku_Monocular_3D_Object_Detection_Leveraging_Accurate_Proposals_and_Shape_Reconstruction_CVPR_2019_paper.pdf | Peer-reviewed conference paper | 以 `t_z=f h/\hat h` 为针孔尺度基线，并显式指出视角、姿态和表观高度差异需要残差修正。 | 以已知物理尺寸和图像尺度构造解析深度，再加入姿态相关低阶修正，具有文献基础。 | Direct at strategy level | 论文修正项不是本文的 `q²` 指数形式。 |
| D2 | Zhu et al., *MonoEdge: Monocular 3D Object Detection Using Local Perspectives*, WACV 2023, https://openaccess.thecvf.com/content/WACV2023/papers/Zhu_MonoEdge_Monocular_3D_Object_Detection_Using_Local_Perspectives_WACV_2023_paper.pdf | Peer-reviewed conference paper | 局部边缘的透视形变比率能够编码目标深度与朝向。 | OBB 长短轴给出的尺度不一致可作为倾斜/透视压缩的观测量。 | Direct at cue level | MonoEdge 使用的 keyedge ratios 与本文 `q` 定义不同。 |
| D3 | Lu et al., *Geometry Uncertainty Projection Network for Monocular 3D Object Detection*, ICCV 2021, https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.pdf | Peer-reviewed conference paper | 使用三维高度、二维投影高度和焦距构造几何深度，并分析二维尺度误差向深度误差的放大。 | 针孔尺度反演应保留几何解释，同时必须承认 OBB 尺寸噪声会被反距离关系放大。 | Direct at geometry level | 其不确定度网络不等于本文的解析校正式。 |
| D4 | Pu et al., *MonoDGP: Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors*, CVPR 2025, https://openaccess.thecvf.com/content/CVPR2025/papers/Pu_MonoDGP_Monocular_3D_Object_Detection_with_Decoupled-Query_and_Geometry-Error_Priors_CVPR_2025_paper.pdf | Peer-reviewed conference paper | 从 `Z_geo=fH/h_bbox` 出发，对视角、形状和姿态引起的几何深度误差建模。 | “解析几何基线 + 几何误差修正”是合理建模范式。 | Direct at strategy level | MonoDGP 的学习式误差先验不能直接证明本文两参数公式。 |
| D5 | Collins and Bartoli, *Infinitesimal Plane-Based Pose Estimation*, IJCV 2014, https://3dvar.com/Collins2014Infinitesimal.pdf | Peer-reviewed journal paper | 平面点的透视对应可用于恢复平面姿态，说明平面各向异性投影包含倾斜信息。 | 四角点/OBB 的长短方向压缩差异具有明确的平面投影几何来源。 | Indirect | IPPE 需要点对应并求完整姿态；本文有意采用更轻的标量校正。 |
| D6 | Criminisi, Reid, and Zisserman, *Single View Metrology*, IJCV 2000, https://ora.ox.ac.uk/objects/uuid%3A9b99df78-df75-49a9-a1b8-1b38e1b173a8 | Peer-reviewed journal paper | 在几何先验成立时，可由单幅透视图像恢复度量信息，并分析测量误差传播。 | 固定抓斗尺寸与已知相机内参可为对象级单目度量提供尺度约束。 | General foundation | 不直接给出本文 OBB 深度公式。 |

## Boundary

本次检索没有发现论文直接提出
`c * Z_S * exp(beta * log(Z_L / Z_S)^2)`。该式是本项目在 D1--D6
所支持的“针孔尺度反演、透视各向异性和几何误差修正”框架下构造的
任务特定二阶模型。`q²` 的偶对称假设及指数化正值约束属于本文推导，
不能写成已有论文的原公式或已被外部数据普遍验证的结论。
