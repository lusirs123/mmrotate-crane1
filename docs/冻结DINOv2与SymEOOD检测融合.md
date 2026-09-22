# 冻结 DINOv2 与 SymEOOD 检测融合

本文是「冻结语义分支 + SymEOOD 旋转检测」主线的唯一维护入口，按研究问题、方法协议、实验演进、当前接口和未解决问题组织。原始材料见文末[来源索引](#appendix-sources)。

## 后续实验记录与当前状态（2026-09-22）

本文件继续作为 DINO 后续实验的统一记录文件，不另建平行实验总账。`docs/archive/20260915_pre_consolidation/` 保留历史记录。以下为本轮工作安排；正文中的历史计划不自动成为当前待办。

### 两个目标

1. **优化检测性能**：改善 DINO 相关检测的正确选框、旋转框定位和连续漏检表现；分别记录独立 DINO 与融合系统的收益和退化，不能混用模型指标。
2. **轻量化**：降低推理延迟、显存和模型规模，同时明确检测性能代价。小型骨干、检测头压缩、教师辅助学生训练等仅为备选方向，尚未选定或启动。

本轮不扩展到可靠性策略优化、深度研究、控制或论文正文写作。评估与数据划分核验服务于以上两个目标，不另立研究主线。

### 当前状态：问题复查完成，正式基线保持不变

- native-S14 候选追踪已在三段固定困难切片的 137 帧上完成重建：136 帧 decode 后含可用候选，历史 NMS 后为 117 帧；失败归因为 NMS suppression 19 帧、ROI regression 1 帧、Top-1 success 117 帧。
- 历史 NMS `0.1` 与已经由 source 固定的正式 NMS `0.5` 配对诊断显示：NMS 后含可用候选帧由 117 增至 128，NMS suppression 失败由 19 降至 8，但 Top-1 仍为 `117/137`；新增保留的 11 帧转化为 final ordering 失败。因此不能继续把 NMS 阈值当作主要优化方向。
- native spatial adapter V3 使用 source train/val 完成 4 epoch 训练，head 峰值显存约 `371.41 MB`；所有 epoch 均丢失旧正确帧，最终 `best_epoch=0`，未替换正式基线。
- 抓料/未完全离料区间 `real_seq03[130,187]` 的 58 帧已完成 measurement-validity 审计。完整 fixed TEST 仍是主结果，排除区间只作为敏感性分析，不能解释为模型改进。
- 当前正式 DINO 组件仍为 native S14、ROI classifier `alpha=0.5`、S7 disabled、NMS IoU `0.5`。native Pairwise V1/V2、连续/相对质量头、high-resolution ranker 和 unified hard-pair 已覆盖小型 OBB 候选质量排序，不能重复实现。优先准备检测性能改进：复用收益/退化证据，只有新增结构或监督依据明确时才开启新训练；缺少独立真实困难验证时不宣称跨域安全接管。具体裁决见下方整体复盘。
- 完整结果、SHA256、分量指标、优化门槛和经原论文核验的文献映射见 [Fixed TEST 测量有效性对照与 DINO 优化路线](measurement_validity_and_dino_roadmap_20260919.md)。

### 2026-09-22 整体复盘：收益、退化与后续改进依据

本节是当前研究决策的补充说明：优先继续准备检测性能改进，轻量化排在有效结构确定之后；运行成本和显存仍随实验记录。上文及历史章节中“所有新训练必须等待新数据”的表述应限定为既有 native/S7 接管路线。数据不足是已确认的证据限制，尚不能推出“所有结构改进均无效”或“增加数据必然解决排序”。本轮不选新 checkpoint，不启动训练，不改变原数据划分或部署门槛。

#### 复盘范围与证据强度

本轮读取统一实验记录的暗光、小目标、排序/时序和 seq11 融合章节，并直接解析下列本地 JSON。历史未取得原始逐帧工件的实验在下文标为“历史记录”，不声称已重算全部实验。

| 原始结果 | 本轮核验内容 | SHA256 |
| --- | --- | --- |
| `work_dirs/dino_teacher_fc_cls_interpolation_v1/source_interpolation_result.json` | 6 个 alpha 的 738 帧逐帧命中转换、分序列 gain/lost、selected alpha | `19e8e4b97fa1cc2c873fb1ba7bcdb7efe17f28ae4ac2d008d3828ec52977055c` |
| `/Users/mac/Downloads/frozen_dino_native_s14_formal_nms_paired_all_v4.json` | 历史与正式 NMS 的成对汇总、三组归因、重建检查 | `82b863f60052607697540b080018f0dac9037f2fa9931a947e3f9e2fba08ccb5` |
| `/Users/mac/Downloads/native_spatial_adapter_training_v3.json` | 4 epoch 指标、具体 gain/lost 帧、epoch-0 回退 | `bd964b39f0ba8dadd2c4a1f6a2aef97b11127f5975b7f702372ef65557cd03f4` |
| `/Users/mac/Downloads/frozen_dino_source_small_coverage_audit_v1.json` | train/val 的 real/sim 小目标覆盖 | `ffe91ad51442ca115410709420a3ede09eaa4ea0649acc5416e89b22ad6fa055` |

这些 SHA 绑定本轮读取的结果文件，不代表已经验证服务器权重字节或重新运行 GPU 推理。Downloads 文件仍在原位；复现时需保留同 SHA 的副本。

#### 首先分清模型和“成功”的含义

- ScopedDINO 暗光救援属于 BrightAug 加 scope 分支；scope 来自已暴露 target，结果只支持诊断。它不能直接作为通用 native DINO 的性能。
- 当前正式 DINO 组件为 native S14、分类权重插值 alpha=0.5、S7 disabled、NMS=0.5。历史 677/738 是绑定评测设置下的 source 命中数，不能跨候选数量、分数阈值、过滤规则直接复用。
- Base V3 是 K1 锚框与 DINO fallback 的融合前端；V5.1 是后续分量观测策略。它们的指标不能证明某个 DINO 排序器成功。
- 候选召回改善、Top-1 改善、几何精度改善、连续性改善、跨序列迁移是不同层面的结果。一个层面通过，不能自动推导其他层面通过。

#### 各路线究竟获得了什么

| 路线 | 正向证据 | 退化/限制 | 本轮结论 |
| --- | --- | --- | --- |
| 冻结 DINO 暗光救援（历史记录） | 暗光 0/33→29/33；因果稳定后 32/33 | scope 由 target 导出；新增输出使 DFR/ACI 统计发生变化；完整序列仍有其他长漏检 | 语义救援有效的机制证据；不能推广为全工况结果 |
| FC 分类权重插值（原始 JSON） | alpha=0.5：662→677，gain=15/lost=0 | 更大更新幅度产生退化；不修复 RPN 覆盖 | 已保留收益，不重复插值 |
| S7 readout / RPN（历史记录） | 第一版 epoch2 small 303→318；后续 small R@100 达64/64 | 第一版 source 丢失23个旧正确帧；高候选覆盖未转成 small Top-1 增益 | 保留候选生成能力；不能直接合并部署 |
| 正式 NMS 配对（原始 JSON） | 可用候选帧117→128 | Top-1仍117；正式归因为8个抑制、11个最终排序、1个回归失败 | 不继续把阈值扫描当主要方案 |
| Pairwise V1/V2（历史记录及本轮代码核查） | V1部分source计数提高 | V2有效pair少，source下降；V1困难small未改善 | 同类分类排序训练已充分尝试 |
| 全局affine、lane仲裁与replay（历史记录） | affine +12/-1；lane v1有更多gain | affine无法区分帧条件；replay令lane调整接近统一+2并增加lost | 少量增益监督被放大，不能解释为条件选择已学会 |
| 连续/相对质量及highres ranker（历史记录） | relative source 691/738、lost=0；highres 688/738、lost=0 | 固定small仍50/64；部分早期版本仅未达绝对门槛 | 具有source有效性，迁移未成立；不等同“质量学习完全无效” |
| unified hard-pair（历史记录） | source696/738、small320/350、lost=1 | 非exact-safe；记录中target诊断未执行 | bounded-risk研究证据，不能当正式最优 |
| 时序ROI projector与接管（历史记录） | 对象检索margin明显改善 | 闭环425/738、lost263；native-anchor最佳663/738、lost31 | 身份相似不等于几何质量；隔离状态仍不能解决错误接管 |
| native spatial adapter V3（原始 JSON） | 个别帧获益、资源可控 | epoch1 +1/-2，epoch2–4 +2/-4，均回退 | 这版残差结构不值得直接延长训练；不能排除所有空间建模 |
| seq11 V4 replay（历史记录） | source有小幅正向，补充数据有作用迹象 | aux-val48帧无困难支持；251帧困难块OOF未完成 | 属于融合支线待验证事项，不是已成立的DINO排序提升 |

#### 已直接复核的收益与退化

从插值 JSON 的每帧 `top1_hit` 重新计算，六个候选均满足 `candidate_hits = 662 + gained - lost`，每个候选有738条记录。

| alpha | full | small | gain/lost | gain：real/sim | lost：real/sim |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 662 | 293 | 0/0 | 0/0 | 0/0 |
| 0.125 | 666 | 297 | 4/0 | 0/4 | 0/0 |
| 0.25 | 671 | 300 | 9/0 | 2/7 | 0/0 |
| 0.5 | 677 | 303 | 15/0 | 5/10 | 0/0 |
| 0.75 | 678 | 303 | 19/3 | 6/13 | 0/3 |
| 1.0 | 679 | 304 | 27/10 | 9/18 | 1/9 |

这证明分类更新有真实收益，也有真实代价。只报679优于677会遗漏10个旧正确帧损失。该表只能解释同一插值实验，不能把27个gain取出来与其他模型结果拼接。

adapter V3的稳定退化帧为 `val|real_seq07|79`、`124`，稳定增益帧为 `162`；epoch2–4额外损失 `val|sim_seq10|339`、`380`，额外获得 `67`。反复发生于同一批帧，支持“稳定的收益/退化取舍”解释；仅凭这些记录尚不能确定是边界特征、回归还是排序导致，需要同帧候选细节才可进一步归因。

覆盖审计直接确认 source-train 有927个小目标帧，其中real446、sim481；source-val的350个小目标帧全部来自sim，real为0。这里“小”取该报告的短边token阈值；不能说验证集整体没有价值，也不能说训练集没有真实小目标。历史支持审计中的“real2033帧native正确、自然S7增益仅sim7帧”属于指定候选与命中协议，不能外推为所有真实训练框的中心/尺度/方向都精确。

#### 可以保留的机制与不能直接结合的部分

1. 保留DINO语义、已验证的分类插值、S7候选覆盖及因果稳定的机制证据。它们作用于不同阶段，有组合研究价值，但现有统一ranker和student已经尝试过部分组合，不能再把简单串联当新方法。
2. 排序错误不只由NMS造成；单帧质量、来源分数可比性、几何细节和跨域监督都可能参与。数据支持不足是已知限制，不是已经证明的唯一原因。
3. 降低连续抖动不能替代正确选框。时序身份学习提升却检测下降，是明确的目标错位证据；不能再直接把cosine当RIoU质量。
4. exact-retention失败要和科学上的净收益分开报告。历史部署门槛继续保留；不能为通过结果临时放宽，也不能把净增益但丢失个别旧正确帧的实验写成毫无价值。
5. 抓料区间分析改变的是评估范围，不是模型能力。完整TEST保留；排除58帧仅作敏感性分析，不能用来消除小目标失败或证明创新。

#### 下一项检测改进的准备门槛

已完成的困难挖掘、NMS追踪、配对损失、paired-view与cross-fit不重做。当前还没有证据足够的新网络方案，本轮不因“必须继续优化”而承诺组合必然提升。

首先复用历史同帧冲突记录；仅当关键字段缺失时，补齐少量source帧的候选证据。统一对照必须绑定相同图像、GT、预处理、候选过滤与基线，记录native/S7来源、候选框、RIoU、分数、NMS命运和最终选择。先检查各实验gain/lost是否重合及是否集中于同一事件；GT选优的互补上界只作诊断，不能成为推理选择器。当前本地未取得全部旧实验的逐帧工件，因此尚不能声称已完成全实验互补矩阵或找到了可部署选择规律。

进入新训练前必须回答：新增输入或监督是什么；与已失败结构哪里不同；可观测的无GT判据如何识别适用情形；在哪个source验证集选模；正常帧退化和显存上界如何控制。无法回答时停止同构训练。新增真实数据是可选的证据补充路径；独立真实困难验证缺失时，结论明确限于现有域，不以同视频拆帧替代未知序列证据。

seq11的251帧困难块OOF属于另一个已有未执行计划：已有block-CV协议，但旧59帧工具不能直接代表251帧V4合同。它只能核验同视频内的融合机制，不能补成未知真实序列验证。仅在明确选择融合支线后推进，不把它作为DINO排序研究的必经步骤。

检测方案取得可复现收益后再系统轻量化；当前只持续记录各阶段耗时与各卡峰值显存。历史DINO前向约1.74–1.98秒/帧，不等于可靠性层耗时，也不能用head内存代替全模型内存。

#### 本轮实际验证与未完成事项

- 已解析上述4份原始JSON、重算插值逐帧gain/lost、核验NMS成对归因与adapter逐epoch保留记录。
- 已阅读实际竞争候选挖掘、质量支持审计、Pairwise V2选模门控及对应测试。6项针对性测试通过，122项未选择；这只验证选定逻辑，不证明GPU全链或训练收益。
- 未训练、未重跑模型推理、未检查服务器checkpoint字节、未执行全套测试。旧27条排序/接管记录和融合实验主要按已有文档复盘；原始工件不齐的结论保留该限制。
- 后续需要的关键材料为旧S7/highres/relative/unified/temporal实验的原始逐帧结果与输入身份；优先定位现有服务器工件，不要求重跑全部历史实验。

#### 帧级互补审计（已完成最小对照，停止 adapter 选择器路线）

仓库新增 `crane_project/tools/dino_quality_ranking_complementarity_audit_v1.py` 与示例契约
`crane_project/data_contracts/dino_quality_ranking_complementarity_audit_v1.example.json`。该入口只读取既有
source-validation JSON，不加载图像、checkpoint 或 GPU，也不训练、不选择新 epoch、不读取 target。它执行以下检查：

1. 所有方法绑定输入路径、SHA256、selector 和提取节点；基线必须含完整逐帧 outcome。
2. 历史方法可用完整 outcome，或使用 `lost_frame_keys`/`gained_frame_keys`；若报告还声明 baseline、retained、lost、gained、candidate 计数，必须与帧列表重算完全一致。
3. 拒绝 target-dependent 结果、未知帧、重复帧、同一帧同时 gain/loss、与基线状态矛盾以及方法名冲突。
4. 输出各方法 Top-1、gain/loss、分序列转换、两两 gain 交并及 GT-dependent union oracle。oracle 只说明方法错误是否互补，明确不授权运行时选择器。

服务器正式结果为 `dino_quality_ranking_complementarity_audit_v1.json`，SHA256
`f0a5b7c2510d4db409d50cedd493e15cd47fc21da31e7acff201ab601c29650e`；Markdown SHA256
`12b7dd0fee408d362c78d0d15c30c8243e59aec263dc6bca749795ec6188e061`。本次最小对照只纳入两份能够绑定同一 `677/738` 基线并提供精确逐帧转换的报告：relative-quality epoch 4 为 `691/738`、`+14/-0`；native spatial adapter epoch 1 为 `676/738`、`+1/-2`。两者 gain 不重合，但相对于最佳单方法 relative-quality，GT oracle 只由 `691` 增至 `692`，即多 1 帧（`0.1355` 个百分点）。该单帧空间不足以继续开发 adapter/relative 运行时选择器，adapter 路线停止，正式 native-S14 `alpha=0.5` 基线不因这项诊断改变。

“只比较两项”不表示历史上只有两项出现过正向结果。本轮排除的主要路线及原因如下：FC alpha `0.75/1.0` 属于同一插值族且伴随旧正确帧损失；highres ranker 历史为 `688/738、lost=0`，但未在本次契约中绑定原始逐帧工件；unified hard-pair 为 `696/738、lost=1`，仅通过 bounded-risk research gate；global affine、lane、S7 readout 及 seq11 属于不同协议、模型身份或数据支线；NMS `0.5` 增加候选保留却没有增加最终 Top-1。它们不能与本次两项结果直接拼成融合成绩。后续若取得 highres/unified 的同基线原始逐帧报告，可扩展只读矩阵，但不为补全矩阵重跑训练或 target 推理。

审计入口已修订为 protocol version 2：除 union oracle 外，必须报告最佳单方法、oracle 相对最佳单方法的新增帧数和百分点，并在契约中记录项目继续门槛。本项目把“相对最佳单方法至少新增 2 帧”设为继续研究选择器的最低资源门槛；该门槛只用于安排研究工作，不参与 checkpoint、阈值或模型选择。按该口径，本次结论为 `COMPLEMENTARITY_BELOW_CONTINUATION_THRESHOLD_STOP_SELECTOR_WORK`。下一步只核对 relative-quality 的方法身份与 14 个收益帧机制，寻找可复用于单模型的质量信息；不再围绕 adapter 的唯一收益帧开发融合。

### 后续单次实验记录格式

每项实验在本节下按日期追加，保留失败结果与未执行状态：

| 字段 | 记录内容 |
| --- | --- |
| 日期、实验编号与状态 | 计划 / 运行中 / 完成 / 停止；不将计划写成结果 |
| 目标与假设 | 性能优化或轻量化；本次具体解决什么问题 |
| 基线与唯一主要改动 | 配置、代码版本、checkpoint 路径及 SHA256；注明独立组件或融合系统 |
| 数据与选择规则 | 训练/验证/测试身份、序列划分、输入绑定和预先确定的选择标准 |
| 实际执行 | 命令、环境、输出目录；未执行部分明确列出 |
| 检测结果 | 沿用绑定报告中的自定义检测指标与定义；报告正确数、几何质量、连续漏检及分组退化，必要时展开候选诊断 |
| 资源结果 | 参数量、显存、延迟；写明硬件、分辨率、批量、预热与计时范围，区分组件和端到端 |
| 结论与证据 | 相对基线的收益/代价、保留或停止理由、结果文件路径与 SHA256、仍待验证事项 |

**本阶段已有只读诊断和 source-only adapter 结果，但没有新的 source-safe 性能改进，正式模型未变。**

**阅读约定**

- 本文档只覆盖「冻结语义分支 + SymEOOD 旋转检测」支线。OBB 观测链（含 Base V3 的真实在线推理入口与固定 TEST 指标链）见 [OBB 观测可靠性与连续输出](OBB观测可靠性与连续输出.md)，深度几何契约见 [Webots 单目深度估计](Webots单目深度估计.md)。**本文档内的「未完成」只在检测融合支线内成立。**
- 正文中的模型身份、数据角色和协议版本是判断数字归属的唯一依据；同一指标只在同一模型、同一协议下可比。
- 第 3 节按实验族记录，每族包含问题、方法变化、冻结条件、数据角色、关键结果、裁决和证据入口。失败实验保留其方法差异与停止条件，不合并为笼统的「均失败」。
- 已被后续裁决取代的计划不再作为待办出现；仅在设计演进需要时保留一句演进说明。
- 「正式部署模型」「当前最值得继续」「下一步只允许」等判断一律带适用支线与记录日期，汇总见 [6.4](#sec-6-4)。
- 本文档只整理已有记录，不表示重新训练、重新推理或复核过权重与外部工件；本轮为语义融合与状态核对，另对 `crane_symeood_dino_unified_v1.py` 做了只读配置核查（见 [6.2.1](#sec-6-2-1)）。

---

## 1. 研究问题与当前结论

<a id="sec-1"></a>

### 1.1 研究问题

<a id="sec-1-1"></a>

研究对象是港口门座式起重机抓斗**顶梁**的单类旋转框（OBB）观测。检测框只包围顶梁参考平面，不包围抓斗平台；单目、单类、纯视觉，不使用 PLC、编码器或其他现场传感器。抓料/未完全离料阶段按统一操作规则标记为 `measurement-invalid`，不能因为某一帧恰好检测成功就保留。

改进目标按困难类型分成三支，它们在数据角色和失败机制上互相独立，不能互相替代：

| 困难类型 | 代表切片 | 失败机制 | 当前状态 |
| --- | --- | --- | --- |
| 暗光大目标 | `test/real_seq02[137..169]`，33 帧 | 几何候选仍存在，分类排序坍塌（正确候选排名极深） | 冻结 DINOv2 语义分支 + 因果框稳定器已恢复 |
| 远距离目标 | `test/real_seq02[2..41]`，40 帧 | 早期候选不足；后经 RPN 逐级审计确认候选实际存在 | 已归因，不再是独立瓶颈 |
| 更小旋转目标 | `test/real_seq03[129..192]`，64 帧 | 短边约 1.124 个 DINO token；候选覆盖已补足，最终 Top-1 排序未解决 | 当前唯一明确未解决瓶颈 |

另有 SymEOOD–DINO 几何融合与 seq11 补充数据一条支线（第 5 节），用于检验「SymEOOD 主头 + 冻结 DINO 候选」能否形成统一的联合检测前端。

<a id="sec-1-2"></a>

### 1.2 当前结论

1. **正式 DINO 组件身份固定**：native S14、ROI classifier 权重插值 `alpha=0.5`、S7 disabled，checkpoint 为 `work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth`。source 指标 full `677/738`、small `303/350`、MCML `3/3`、旧正确帧丢失 `0`；DINO ROI NMS IoU 固定为 `0.5`。该 `alpha=0.5` 是**分类器权重插值系数**，与早期暗光分支因果框稳定器的 `alpha=0.25`（尺度/角度 EMA 系数）含义不同，两者不可互换。
2. **暗光大目标已有可用机制证据**：在 `real_seq02[137..169]` 上，BrightAug top-1 为 `0/33`、RIoU-MCML 为 `33`；ScopedDINO 为 `29/33`、MCML `1`；加入因果框稳定器后为 `32/33`、MCML `1`。该切片参与了问题定位与 scope 设计，因此只是 target-dev 机制诊断，不是独立 zero-shot final test。
3. **小目标候选覆盖已解决，最终排序未解决**：高分辨率 S7 把固定 `seq03_small` 的 R@100 / usable geometry 从 `55/64` 提高到 `64/64`，但 Top-1 仍为 `50/64`。候选接管、框参数平滑、分量有效性管理是三种不同机制，不能合成一个含糊的「时序优化模块」。
4. **安全约束是部署前提，不是可放宽的超参数**：affine、lane、static/unified ranker 等实验族的 source 原始收益都伴随旧正确帧丢失（`lost>0`），而满足 exact retention 的版本又没有 small Top-1 增益。
5. **learned S7 分支的门槛状态必须分项记录**，不能概括为「全部失败」：
   - protocol-24 的 stride-7 ranker（epoch 3 + runtime margin `0.225`）通过了**正式 exact-retention source gate**（full `688/738`、small `311/350`、`lost=0`、MCML `3/3`）；
   - protocol-26 的 unified hard-pair ranker 通过了**bounded-risk research gate**（full `696/738`、small `320/350`、`lost=1`、lost fraction `0.001477`），该门不授权 source-safe 声明；
   - 两者都**没有通过三段 fixed target-dev**（`seq03_small` 仍为 `50/64`），因此正式部署状态回退到 native S14 `alpha=0.5`、S7 disabled，`best_epoch=0`。
   详见 [3.2.4](#sec-3-2-4) 的门槛汇总表。
6. **当前根因是监督与验证支持不足**：既有 source split 中「native 错误而 S7 正确」的自然增益样本极度稀疏（source-train 仅 7 帧，且全部位于 `sim_seq08`），同时不存在可用的 real 困难留出序列，因此无法从现有 source 数据训练并证明一个跨域安全的质量排序/接管器。
7. **联合前端尚未验证**：`crane_symeood_dino_unified_v1.py` 已实现 SymEOOD 主头与冻结 DINO 候选的联合推理结构，但只是新的、尚未验证的组合模型；SymEOOD 与 DINO 各自的数字只能作为组件证据。
8. **未闭环项按支线区分**（某个配置未完成不等于整个项目未完成）：
   - **Base V3 观测链（OBB 主线）**：图像→OBB 的真实在线推理入口与固定 TEST 指标链已实现并冻结为报告，见 [OBB 观测可靠性与连续输出](OBB观测可靠性与连续输出.md) 第 4 节；该支线未完成的是未知新序列与物理状态证据。
   - **DINO/S7 支线**：统一训练/推理入口、自包含最终 checkpoint、蒸馏与实时化、真正未知序列泛化、模型冻结后的完整 final test 未完成，且完整 test 未获授权。
   - **联合前端 `unified_v1`**：仅实现、未验证，联合 source gate 尚未运行。
   - **检测融合 seq11 支线**：251 帧三折时序块 OOF/CV 未执行。
9. **正式 NMS 只改善候选保留，不改善最终 Top-1**：在同一 137 帧候选追踪集合上，NMS IoU `0.1 → 0.5` 使 NMS 后存在可用候选的帧数 `117 → 128`，但 Top-1 命中保持 `117`；11 帧由 NMS suppression 转成 final ordering 失败。后续重点应从继续调 NMS 转向候选级 OBB 几何质量排序。
10. **native spatial adapter V3 已停止**：四个 source-only epoch 均未通过旧正确帧零丢失门，最优身份回退 `epoch=0`；该实验只证明 stride-14 轻量残差 adapter 可以稳定训练和控制显存，不能证明检测提升。
11. **抓料阶段只改变报告分层**：`real_seq03[130,187]` 共 58 帧的 measurement-validity 审计不参与训练、阈值或 checkpoint 选择。完整 fixed TEST 是主结果，measurement-valid 只作为敏感性结果并列报告。

<a id="sec-1-3"></a>

### 1.3 模型身份对照

以下名称在历史记录中经常混用，必须按本表区分。同一份结果只能归属其中一个身份。

| 身份 | 定义 | 数据角色 / 当前资格 |
| --- | --- | --- |
| **ScopedDINO（暗光语义救援分支）** | scope-gated frozen-DINO semantic rescue branch：BrightAug 主检测器 + 冻结 DINOv2 ViT-L/14 语义路径 + source-only 训练的单尺度 oriented RPN/ROI 小头 + `alpha=0.25` 因果框稳定器。低照度 scope 由 target-dev 困难切片导出（`target_label_derived=true`）。 | 暗光大目标的机制证据来源；完整 test 结果标记为 diagnosis-only，不是 zero-shot 泛化声明。 |
| **正式 native-S14 DINO 组件** | 冻结 DINOv2 native S14 + ROI classifier `alpha=0.5` + S7 disabled，checkpoint `source_safe_interpolated_head.pth`。 | 当前唯一正式 DINO 组件与部署基线；也被 Webots 深度接口与联合 config 引用。 |
| **Base V3 epoch9** | SymEOOD K1 epoch24 有框时作为锚框、K1 缺失时使用冻结 DINO 候选的融合检测前端。refiner 输出的固定分数 `1.0` 是常数，不是检测置信度。 | OBB 观测可靠性主线采用的冻结前端（见 [OBB 观测可靠性与连续输出](OBB观测可靠性与连续输出.md)）。 |
| **V4+seq11-v2 replay epoch10** | Base V3 epoch9 初始化 + retention loss + 原始 2781 帧 replay + seq11 train 203 帧（合计 2984 帧）训练的 refiner。 | **历史暂定 source-retention candidate**，`source_gate_passed=False`，缺少困难块 OOF/CV 支持，不能写成已验收最终模型。 |
| **普通 SymEOOD K1 epoch24** | config `crane_project/configs/crane_symeood_k1_source_val_eval.py`，checkpoint `work_dirs/crane_symeood_k1/epoch_24.pth`，SHA256 `57233e5423de4a9d0a67fd51058cd7d92adaac5d6621f275cc0ec5b0fc7f9ee2`。 | 与 V4+seq11-v2 不是同一模型；不得用其它模型指标替代其身份。 |
| **BrightAug（简单数据增强）** | 增强管线只含 `RRandomFlip`（水平/垂直/对角各 0.25）+ `RandomBrightnessContrast`（仅 brightness 0.4–1.0，prob 0.5），没有随机裁剪、旋转或对比度扰动。checkpoint `work_dirs/crane_symeood_k1_brightaug/epoch_20.pth`。 | ScopedDINO 的通用检测器底座；暗光分支之外的帧完全沿用其结果。 |

<a id="sec-1-4"></a>

### 1.4 阅读与证据边界

- **证据层级分层**，不混图、不共享阈值选择：
  1. `source-train`（official train + train_sim，2781 帧）：只用于训练。
  2. official `source-val`（738 帧，`real_seq07` + `sim_seq10`）：用于模型选择与 source gate。后期可靠性工作中该 split 已被当作开发数据使用。
  3. seq11 同视频辅助验证（48 帧）：只能验证同一视频内的机制支持，不能证明未知序列泛化。
  4. 三段 fixed target-dev：`seq02_far`（40 帧）、`seq02_dark`（33 帧）、`seq03_small`（64 帧）。它们参与了问题定位和方法设计，只能作为固定诊断集合，**不能拼成一个模型的总成绩**，也不是未知序列或部署测试。
  5. 完整 fixed TEST（992 帧）：**已暴露**，只用于固定模型的报告与失败归因；不据此重开训练、特征或阈值选择。
  6. 未知独立视频：当前没有。
- 本文档中所有 target 数字均为历史上的 `source-trained, target diagnosis-only` 结果。
- 旧 scope 定义、域路由开关和旧实验命令是历史记录；只有第 5 节明确列出的入口是当前可复现入口。

---

## 2. 方法、符号与必要协议

<a id="sec-2"></a>

### 2.1 符号约定

<a id="sec-2-1"></a>

第 `t` 帧最终选中的旋转框记作

```text
b_t = (c_x,t, c_y,t, w_t, h_t, theta_t, score_t)
```

因果框稳定器只保留上一帧的稳定化几何状态：

```text
hat_b_(t-1) = (c_x,t-1, c_y,t-1, hat_w_(t-1), hat_h_(t-1), hat_theta_(t-1))
```

主要评价量：

| 指标 | 定义 |
| --- | --- |
| `top1_hits` | 最终第一名检测框满足 `RIoU >= 0.5` 的帧数 |
| `top1_mcml` / MCML | 连续 top-1 失败的最长帧数 |
| `RPN R@K` | 前 K 个 RPN proposal 中存在 `RIoU >= 0.5` 候选的比例 |
| `decoded_geometry_exists` | 经 ROI bbox regression 后是否仍存在可用几何 |
| `post-NMS R@K` | NMS 后候选是否保留 |
| source exact retention | 正式 DINO 原先正确的 source 帧是否全部保留（`lost=0`） |
| `R_center^det@15px` | 有输出帧的条件中心精度，只统计有输出的帧 |
| `R_center^all@25px` | 含 silence 的全帧中心召回率（real 用 25px、sim 用 10px，silence 记为失败） |

两套 `R_center` 含义不同，论文中不能写成同一个无说明的 `R_center`。

<a id="sec-2-2"></a>

### 2.2 冻结 DINOv2 语义救援分支

**核心结构**

```text
输入图像
  ├─ BrightAug 主检测器：通用路径，保持原有能力
  └─ 冻结 DINOv2 ViT-L/14 patch grid
       -> 单尺度 OrientedRPNHead
       -> RotatedRoIAlign 7x7
       -> RotatedShared2FCBBoxHead
       -> 旋转框候选
       -> source-selected 因果旋转框稳定器
            仅平滑宽、高和周期角度
            保持中心、分数、排序和输出存在性不变
```

DINOv2 主干不更新，只训练 RPN/ROI 检测头。推理时不做 score fusion：scope 内 DINO 是 primary，DINO 无输出时才回退 BrightAug；scope 外完全使用 BrightAug。因果稳定器只处理 scope 内最终选中的 top-1 旋转框，不重新选择候选。

设计借鉴 CVPR 2025 DINO Teacher 的核心思想：用冻结的 DINOv2 语义表征作为教师式视觉基础，再训练轻量检测头适配任务。本项目实现的是一个受控的、source-only 的冻结语义分支，**没有**复现原论文的 target pseudo-label、教师-学生联合训练或完整域对齐流程：DINOv2 权重冻结；检测头只用 source 标注训练；不使用 target pseudo label；不在 target 上做 optimizer step；target 标签只用于诊断和离线评测。

**为什么需要因果稳定器**

冻结 DINOv2 分支解决的是「目标能否被找到并排到第一位」，但 DINO 的 RPN/ROI 检测头逐帧独立预测。暗光片段恢复输出后，相邻帧的框宽、高和角度可能跳动，使 DFR 上升、ACI 下降。这不是 DINO 找错了目标，而是同一个目标在不同帧使用了不稳定的旋转框参数。因此方法拆成两个互补步骤：

```text
冻结 DINOv2 语义分支：恢复目标发现和分类排序
因果旋转框稳定器：    约束已恢复目标的跨帧尺度与角度变化
```

稳定器不参与候选生成和分类，也不改变 DINO 与 BrightAug 的路由决策。它只使用当前框与上一帧状态，不读取未来帧，因此是严格在线因果处理，而不是双向平滑或 teacher-forced 后处理。它明确保持以下量不变：

```text
hat_c_x,t = c_x,t
hat_c_y,t = c_y,t
hat_score_t = score_t
候选顺序、检测是否存在、DINO/BrightAug 路由均不改变
```

**旋转框等价表示对齐**

同一个旋转矩形有两种等价参数：`(w, h, theta) == (h, w, theta + pi/2)`。直接对两种表示做平均，可能把一个长方形错误地平滑成接近正方形并产生约 `90°` 的虚假角度跳变。平滑前先在当前框的两个等价表示中选择与上一稳定框最接近的一个，代价为

```text
C = || log([w,h]) - log([hat_w_(t-1),hat_h_(t-1)]) ||_2^2
    + Delta_pi(theta, hat_theta_(t-1))^2
```

其中 `Delta_pi` 是考虑 `pi` 周期后的最短角度差。选择代价较小的表示进入后续平滑；这一步只消除参数表达歧义，不改变旋转矩形的实际几何区域。

**尺度和周期角度平滑**

宽、高在对数空间使用因果指数滑动平均：

```text
log(hat_w_t) = (1-alpha) log(hat_w_(t-1)) + alpha log(w_t)
log(hat_h_t) = (1-alpha) log(hat_h_(t-1)) + alpha log(h_t)
```

使用对数空间是因为尺度变化更接近比例变化，且能保证平滑后宽高始终为正。角度具有 `theta` 与 `theta+pi` 等价的周期性，不能直接线性平均，先映射到双角单位圆：

```text
v_t     = [cos(2 theta_t), sin(2 theta_t)]
hat_v_t = (1-alpha) hat_v_(t-1) + alpha v_t
hat_theta_t = 0.5 atan2(hat_v_t[y], hat_v_t[x])
```

双角表示把相差 `pi` 的两个等价方向映射到同一点，能跨越角度边界平滑，避免 `89°` 与 `-89°` 被错误平均到 `0°`。

**系数选择和状态重置**

候选系数固定为 `alpha in {0.25, 0.5, 0.75, 1.0}`，`alpha` 越小历史状态影响越强，`alpha=1.0` 表示完全不修改当前框（保底项）。系数只在 official source validation 上选择，target 标签不参与。候选必须同时满足：source-val top-1 不下降、MCML 不增加、mean RIoU 下降不超过 `0.005`、ACI 不下降；在合格候选中选择 DFR 最低者，最终得到 `alpha=0.25`（保留 75% 上一稳定状态、吸收 25% 当前观测）。出现以下任一条件时立即清空状态：

- 当前帧不在 DINO scope 内；当前帧没有检测输出；序列发生变化；帧号不连续。

因此稳定器只在同一低照度片段的连续有效输出之间工作，不影响 scope 外 BrightAug 的检测结果。

<a id="sec-2-3"></a>

### 2.3 顺序推理流程与正式代码位置

最终实现不再依赖先生成多个 pickle 再用离线工具拼接，正式 config 直接构建 `ScopedDinoLowlightDetector`：

```text
当前帧
  -> BrightAug 主检测器
  -> 读取预定义低照度 scope
       scope 关闭：原样返回 BrightAug，并清空时序状态
       scope 开启：
         -> 冻结 DINOv2 ViT-L/14
         -> source-only Oriented RPN/ROI head
         -> valid-content 过滤
         -> DINO top-1；无 DINO 输出时回退 BrightAug
         -> alpha=0.25 因果框稳定器
              中心、分数、排序、输出存在性不变
              只平滑 log(w)、log(h) 和 pi 周期角度
  -> MMRotate 标准一类别检测结果
```

时序状态在 scope 关闭、检测 silence、序列变化或帧号不连续时清空，不会跨序列、跨困难片段传播历史框。

```text
组合 detector:
  mmrotate/models/detectors/scoped_dino_lowlight_detector.py
唯一正式 config:
  crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py
完整论文指标入口:
  mmrotate/datasets/crane_custom_dota.py
  crane_project/tools/eval_crane_offline.py
```

`CraneDataset.evaluate()` 由 config 的 `paper_temporal=True` 启用完整指标：一次标准 `tools/test.py` 同时输出标准 mAP、全帧中心召回、`R_center^det@15px`、`R_center^all@25px`、DFR、ACI、TDR、MCML、MRF 和 mean RIoU。旧的 audit/tool 文件保留为论文证据和回归检查，但不再承担最终推理组合。

部署方式（三张 8 GB GPU）：

```text
GPU 0：BrightAug + 冻结 rotated RPN/ROI head
GPU 1：DINOv2 前 12 个 transformer blocks
GPU 2：DINOv2 后 12 个 transformer blocks
```

BrightAug 和 DINO head 按帧顺序执行，不同时保留两套前向激活；进入 DINO 前主动释放 BrightAug CUDA cache。该模型只支持非分布式、batch size 1 的顺序推理，因为因果稳定器需要严格帧序和单一状态流。

<a id="sec-2-4"></a>

### 2.4 数据协议与证据层级

**source-only 训练与选择**

```text
source train：train + train_sim，共 2781 帧
source val：  official val，共 738 帧（real_seq07 + sim_seq10）
source-small：按 source token 尺度定义，共 350 帧
```

所有新增训练实验均满足：

```text
DINOv2 frozen = True
target_used_for_training = False
target_used_for_checkpoint_selection = False
```

**target 只用于训练完成后的诊断**

```text
seq02_far：  test/real_seq02[2..41]，40 帧
seq02_dark： test/real_seq02[137..169]，33 帧
seq03_small：test/real_seq03[129..192]，64 帧
```

这些切片已参与问题定位和方法设计，不能用于 checkpoint 或超参数选择，也不能称为完全未观察的 final test；正确名称是 `source-trained, target diagnosis-only`。

**seq11 补充数据角色**

251 帧均采用 `k0=1.9`、top-beam-only 标注；manifest 划分为 train 203 / aux-val 48，文件 overlap 0。48 帧辅助验证不覆盖困难样本，不能提供「seq11 困难机制有效」的证据。

<a id="sec-2-5"></a>

### 2.5 训练设置与证据支持范围

以下设置按支线分别说明，只适用于其来源实验。**没有来源或配置支持的内容不作为项目通用规则**；本节末尾单列逐条说明哪些说法曾被写成通用规则但不成立。

<a id="sec-2-5-1"></a>

#### 2.5.1 DINO 小头（暗光语义分支）：8 epoch 独立日程

DINO 小头的 epoch 与 BrightAug 的 epoch 没有对应关系。当前正式协议为：

```text
epochs：8
optimizer：SGD；lr：0.001；momentum：0.9；weight decay：1e-4
gradient clip：10；linear warmup：1000 iterations，ratio=0.001
step decay：epoch 5、7，gamma=0.1
checkpoint：每个 epoch 保存并在 source-val 评估
seed：0
```

source-val 选择结果：best DINO head epoch `8`、source-val top-1 `662/738`、MCML `7`、R@20 `662/738`、mean top1 RIoU `0.6592`。该日程由 `dino_teacher_rotated_labeller.py` 的调用参数固定，仅适用于暗光 DINO 小头本身。

<a id="sec-2-5-2"></a>

#### 2.5.2 DINOv2 与可训练头配置

```text
模型：DINOv2 ViT-L/14；patch size：14
输入高度：600；最长边：1333
DINO blocks：GPU 1、GPU 2 分片；RPN/ROI head：GPU 0
PyTorch 兼容：legacy SDPA，query chunk=512
OrientedRPNHead：单尺度，stride=14，feat_channels=256
anchor scales：32、64、128、256、512 像素；anchor ratios：0.5、1.0、2.0
RotatedRoIAlign：7x7；RotatedShared2FCBBoxHead：fc=1024，单类别
ROI samples：256；proposal count：2000；max detections：2000
```

该配置在 S7 / 时序 / 接管实验族中作为公共块复用（见 [5.3(b)](#sec-5-3)）。

<a id="sec-2-5-3"></a>

#### 2.5.3 检测器侧的初始化与选模（按支线）

| 支线 | 初始化与选模 | 来源 |
| --- | --- | --- |
| BrightAug | 使用独立的 source-val `ckpt_sweep.py` 选择流程，候选 epoch 为 `16、18、20、22、24`；ScopedDINO 的完整 test 使用 `epoch_20.pth` | 来源 L 第 4.4 节 |
| SymEOOD K1（普通） | 作为检测融合支线的冻结底座，使用 `work_dirs/crane_symeood_k1/epoch_24.pth`；因果历史 refiner 的 smoke 明确使用它而**不是** BrightAug `epoch_20.pth`（记录中把误用 `epoch_20` 列为已修复问题） | 来源 F 第 8.2 节、第 12 节 |
| V4+seq11-v2 replay | Base V3 epoch9 初始化并作为 teacher | 来源 F 第 11.2 节 |
| 联合前端 `crane_symeood_dino_unified_v1.py` | 配置层指定的 SymEOOD 位置权重与原始记录不一致，见 [6.2.1](#sec-6-2-1) | 配置核查（本轮） |

`--seed 0` 由仓库 `tools/dist_train.sh` 固定传入，labeller 命令与锁定审计入口也显式使用；V4 replay 的 work dir 名携带 `seed3407`。除这些已记录位置外，本文档不主张其它随机性设置。

S7 / 时序 / 接管实验族复用 `dino_teacher_scoped_lowlight_v1_formal8/feature_cache`（见 [5.3(b)](#sec-5-3)）；跨视图审计使用带 view 名称与版本的 cache key。各实验使用各自的 work_dir（见 [5.2](#sec-5-2)）。

<a id="sec-2-5-4"></a>

#### 2.5.4 曾被写成「统一实验纪律」但缺乏来源支持的说法

| 说法 | 核查结果 |
| --- | --- |
| 「新分支统一从 `work_dirs/crane_symeood_k1/epoch_24.pth` 初始化，再训练 24 epoch」 | **不成立为通用规则。** 全部 config 的 `load_from` 均为 `None`，初始化由命令行 `--cfg-options load_from=...` 或 wrapper 在运行时指定；`epoch_24.pth` 是检测融合支线的冻结底座，不是全项目统一初始化。V4 replay 使用的是 Base V3 epoch9 初始化 |
| 「checkpoint 只按 `16 18 20 22 24` 串行比较」 | 该候选集只属于 BrightAug 的 `ckpt_sweep` 选择流程；不适用于 DINO 小头（8 epoch）与各 S7 实验（4 epoch / 按协议锁定） |
| 「根目录 `dist_train.sh` 固定 `--seed 0`」 | 已在本仓库脚本中核实（`tools/dist_train.sh` 传入 `--seed 0`），但归档文档未记录该脚本；仅对走该脚本的 MMRotate 训练成立，labeller 入口的 `--seed 0` 是另一处 |
| 「代码或 checkpoint 改动后必须使用全新 work_dir / 缓存」 | 归档只有「本轮训练使用了新的服务器目录」这类对单次运行的记述，没有形成协议条款。本节改述为各实验使用各自 work_dir 的事实 |
| 「smoke 只判断数值和接口健康，不能替代路线结论」 | 归档记录的是具体 smoke 结论（如 `ALLOW_K1_RETENTIVE_SEQ11_V2_REPLAY_SOURCE_TRAINING`、因果历史 refiner smoke 通过），它们是入口校验输出，不构成性能证据；本文档按此理解使用，不作为独立结论引用 |

<a id="sec-2-5-5"></a>

#### 2.5.5 通用 source gate 检查项

任何新方法的 source gate 都必须显式检查：旧正确帧 `lost=0`、full/small 绝对 Top-1、MCML、DFR/ACI 不退化；source gate 未通过不读取 target。个别实验在预注册时另加多序列净增益等条件，见 [3.2.4](#sec-3-2-4) 对应记录。

---

## 3. 实验结果及演进裁决

<a id="sec-3"></a>

第 3 节各实验族保留完整的方法差异、冻结条件、关键数据、失败原因和停止条件。所有实验结果都是 `source-trained, target diagnosis-only`，除非明确标注为 source-val 或完整 test。历史 target 数字只能在同一方法、同一 checkpoint 和同一 gate 下比较，不得跨实验拼接为单一模型结果。

<a id="sec-3-1"></a>

### 3.1 暗光大目标：语义排序恢复

<a id="sec-3-1-1"></a>

#### 3.1.1 问题与根因

针对 `real_seq02[137..169]` 暗光大目标困难片段的**分类排序坍塌**，不是普通亮度增强问题，也不是远距离候选生成修复。对 source 训练、target atlas、FPN pathway 和分类卷积的联合审计得到：

```text
source 稳定路径：FPN level1 + anchor0 + classifier0
target 暗光正确路径：FPN level0 + anchor2 + classifier2
```

source 的 anchor2 没有有效的正几何和分类监督；target 中大多数困难帧仍存在 `RIoU >= 0.5` 的候选，正确候选只是被分类排序推到很深的位置。因此问题不能描述为「暗光导致所有特征变弱」，准确描述是：

```text
geometry remains
classification ordering collapses
```

已排除的简单解释（各自对应一次受控审计，结论是这些方向都不能单独恢复 top-1）：

| 已排除的解释 | 证据 |
| --- | --- |
| padding / valid-content 校正 | 只改善 usable rank，不能恢复 top-1 |
| quality-only teacher-forced | 只能达到 oracle 上界，不能转化为可训练部署收益 |
| 复制 source anchor0 filter | 不能识别 target level0 特征，target top-1 仍 `0/33` |
| guided background proxy | 同时破坏分类和几何，不能授权 adapter |
| AdaBN 统计偏移 | 统计偏移没有被亮度条件稳定区分，不能作为已验证根因 |
| P3 objectness、P3/P4 邻域、普通 FPN kNN | 没有提供普通特征空间迁移证据 |
| 普通多尺度 FPU 邻域救援 | P3/P4 均未通过 source-control gate |
| 亮度增强（含 Retinex 类） | 属于通用增强路径，不能解决分类路由错配 |

`real_seq02[2..41]` 的远距离/候选饥饿问题与 `137..169` 的暗光排序问题分开处理，不合并成功率。

<a id="sec-3-1-2"></a>

#### 3.1.2 结构性诊断阶段（正式训练 DINO 小头之前）

1. **Anchor/FPN coverage**：确认 source 的稳定监督集中于 level1/anchor0，target 暗光正确候选位于 level0/anchor2。
2. **Shared-filter counterfactual**：复制 anchor0 filter 后 target top-1 仍为 `0/33`，排除「只需增强 anchor0 filter」。
3. **Frozen P3 objectness probe**：source-val 很高，但 target paired margin `0/31`，不能支持普通 P3 objectness 迁移。
4. **Feature alignment audit**：target usable P3 特征在 source whitening、方向和 cosine 审计中全部呈 source-negative-like，强支持 level/feature domain shift 解释。
5. **P3/P4 neighborhood rescue 与 multimodal kNN**：均未通过 source-control gate，关闭普通 FPN 邻域救援路线。
6. **ROI head probe**：冻结 DINO 特征上的 source ROI head 在 source-val 可以稳定学习，早期 target 结果不足以作为正式部署结论。
7. **Temporal beam / rerank**：只能把少量候选提升到 top-1，不能稳定解决整段失败，关闭为主要方法路线。

**历史 legacy 诊断（不作为正式协议）**：早期曾使用 `val/real_seq07` 按 `frame_id % 5` 划分（181 帧 source-train、45 帧 source-val），得到过 target-dev `32/33、MCML=1`。该结果用于路线探索，但没有使用 official train/train_sim 与 official val 的正式划分，正式论文结果应使用 2781 帧 source train、738 帧 source val、8 epoch 的正式 source-only 训练结果。

<a id="sec-3-1-3"></a>

#### 3.1.3 暗光专项结果

```text
片段：real_seq02[137..169]

方法                         top-1       旋转 IoU-MCML
BrightAug                    0/33        33
ScopedDINO（正式 source-only）29/33       1
稳定化 DINO                  32/33       1
```

该结果说明冻结 DINOv2 的语义特征能够绕开 BrightAug 的暗光分类路由失配，恢复了暗光大目标的分类排序，而不是简单增加随机背景框。

作用范围（完整 test，992 帧）：

```text
总帧数：992
DINO scope：33 帧
BrightAug-only：959 帧
target-dev scope：target_label_derived=True
final-test eligible：False
```

因此 `sim_seq09` 和 `real_seq03` 没有被 DINO 改动，普通片段的 BrightAug 能力得到保留。

<a id="sec-3-1-4"></a>

#### 3.1.4 完整 test 结果

保留项目一直使用的 `CraneOfflineEvaluator` 输出格式，便于与历史实验直接比较。三列分别是 BrightAug、+冻结 DINOv2、+冻结 DINOv2+因果框稳定器。

| 域 / 层级 / 指标 | BrightAug | +冻结 DINOv2 | +因果框稳定器 |
| --- | ---: | ---: | ---: |
| real 第一层 `R_center` (%), 15px 条件 | 98.91 | 97.39 | 97.39 |
| real 第一层 `mean_RIoU` | 0.8251 | 0.8214 | 0.8212 |
| real 第二层 `DFR` (%/frame) | 2.6441 | 3.8323 | 2.6402 |
| real 第二层 `ACI` | 0.9364 | 0.9182 | 0.9395 |
| real 第三层 `TDR_w10` (%) | 80.85 | 88.31 | 88.31 |
| real 第三层 `MCML_max` / `MCML_mean` / `MCML_pass(limit=5)` | 39 / 25.5 / 0 | 39 / 25.5 / 0 | 39 / 25.5 / 0 |
| real 第三层 `MRF` (frames) | 1.0 | 1.0 | 1.0 |
| sim 第一层 `A-RMSE` (deg) | 1.5488 | 1.5488 | 1.5488 |
| sim 第一层 `R_center` (%) | 100.0 | 100.0 | 100.0 |
| sim 第一层 `mean_RIoU` | 0.9204 | 0.9204 | 0.9204 |
| sim 第二层 `DFR` (%/frame) | 1.8935 | 1.8935 | 1.8935 |
| sim 第二层 `ACI` | 0.9621 | 0.9621 | 0.9621 |
| sim 第三层 `TDR_w10` (%) | 100.0 | 100.0 | 100.0 |
| sim 第三层 `MCML_max` / `MCML_mean` / `MCML_pass` | 0 / 0.0 / 1 | 0 / 0.0 / 1 | 0 / 0.0 / 1 |

`CraneDataset` 标准评测（必须与上面的 legacy 离线指标分开命名）：

| 指标 | BrightAug | ScopedDINO | 稳定化 DINO |
| --- | ---: | ---: | ---: |
| standard mAP | 0.81683 | 0.81705 | 0.81750 |
| `real_R_center`（全 real 帧） | 0.6476 | 0.7238 | 0.7238 |
| `sim_R_center` | 1.0000 | 1.0000 | 1.0000 |
| `Weighted_R_center` | 0.8943 | 0.9171 | 0.9171 |
| `sim_A_RMSE` | 1.5487 | 1.5487 | 1.5487 |

两套 `R_center` 含义不同：`CraneOfflineEvaluator` 当前完整 test 调用使用 `center_thresh=15px` 且只对有输出的帧统计，所以 `98.91%` 是条件中心精度；`CraneDataset` 使用配置中的 real `25px`、sim `10px`，silence 记为失败，所以 `0.6476 -> 0.7238` 是全 real 帧中心召回率。

**完整 test 的正确解释**：ScopedDINO 把精确旋转 IoU top-1 命中数从 `809/992` 提升到 `838/992`，但完整 real 域 `MCML_max` 仍为 `39`。原因是：

- `real_seq02` 的 `2..40` 是远距离候选不足问题，DINO scope 不覆盖；
- `real_seq03` 的 `129..192` 是小目标几何/候选问题，DINO 没有启用；
- 暗光排序失败区间 `133..171` 被打散，DINO 的局部收益被完整序列 MCML 掩盖。

<a id="sec-3-1-5"></a>

#### 3.1.5 DFR/ACI 退化与因果稳定器的修复

未稳定化的 DINO 版本存在时序几何退化：

```text
real DFR: 2.6441 -> 3.8323
real ACI:  0.9364 -> 0.9182
```

只读框稳定性审计确认，ScopedDINO 的 DFR 退化主要来自其在 `real_seq02[137..169]` 新恢复的连续输出，而不是 scope 外结果被改写。据此新增一个不训练模型的 `Source-Selected Causal DINO Box Stabilization Probe`，它只对 scope 内 top-1 DINO 框的宽、高和角度进行因果 EMA，不改变检测是否存在、候选顺序与分数、框中心，以及 scope 外任何检测结果。

```text
real_seq02 output-stream DFR: 2.1275 -> 4.5157
共同有输出转移 DFR:          2.1275 -> 2.2816
新增可观测转移:              31
scope 外变化帧:              0
```

原因是 DINO ROI head 逐帧独立预测，暗光片段中预测框的宽、高和角度明显跳动；同时 DINO 填补了 BrightAug 原先的 silence，使更多相邻框变化进入 DFR 统计。只使用 source validation 选择的因果稳定器（结构见 [2.2](#sec-2-2)）修复了该问题：

```text
ScopedDINO real DFR:       3.8323 -> 2.6402
ScopedDINO real ACI:       0.9182 -> 0.9395
BrightAug real DFR/ACI:    2.6441 / 0.9364
稳定化后 top-1/MCML:       32/33 / 1
scope 外变化帧:            0
```

系数只在正式 source validation 上按「source-val top-1 不下降、MCML 不增加、mean RIoU 下降不超过 `0.005`、ACI 不下降」四条件筛选，再在合格候选中选 DFR 最低者，得到固定 `alpha=0.25`，随后一次性用于完整 test。target 标签只用于事后诊断。

<a id="sec-3-1-6"></a>

#### 3.1.6 数据泄露审计与声明边界

**没有发生的泄露：** target-dev 没有进入 DINO 小头训练；没有参与 source checkpoint 选择；没有 target optimizer step；没有 checkpoint 写入 target-derived 权重；DINOv2 与 RPN/ROI head 参数在完整 test 前后保持不变；959 个非 scope 帧完全沿用 BrightAug。

**仍然存在的声明限制：** `real_seq02[137..169]` 参与了困难问题定位、scope 设计和方法验证，因此不能称为完全未见的独立 final test。论文应分开报告：

1. source-val：用于 DINO head checkpoint 选择；
2. target-dev low-light hard subset：证明暗光大目标排序恢复；
3. full test：报告整体组合效果和非回归情况；
4. `real_seq02[2..41]`、`real_seq03[129..192]`：作为独立的远距离/小目标失败分析，不与暗光排序问题合并。

当前 scope manifest 明确标记 `target_label_derived=true`，代码已把这一限制固化在 manifest 中，没有把 target 标签写入权重。

<a id="sec-3-1-7"></a>

#### 3.1.7 论文可用表述与创新点定义

**方法效果**：为缓解暗光大目标场景中的分类排序坍塌问题，引入冻结 DINOv2 ViT-L/14 语义分支，并仅使用 source 标注训练轻量旋转 RPN/ROI 检测头。BrightAug 作为通用检测器保留，DINO 分支只在预定义的低照度困难范围内启用，二者不进行分数融合。

**暗光专项结果**：在 `real_seq02[137..169]` 暗光大目标困难切片上，BrightAug 的 top-1 命中率为 `0/33`，冻结 DINOv2 分支提升至 `29/33`，旋转 IoU 连续最大失败长度由 `33` 降低到 `1`。

**完整 test 结果**：在 992 帧完整测试集上，稳定化 ScopedDINO 将全帧 real 中心召回率从 `0.6476` 提升至 `0.7238`，Weighted center score 从 `0.8943` 提升至 `0.9171`，mAP 从 `0.81683` 提升至 `0.81750`；real DFR 从未稳定化 DINO 的 `3.8323%/frame` 降至 `2.6402%/frame`，ACI 从 `0.9182` 提升至 `0.9395`；sim 域全部指标保持不变。

**限制**：低照度 scope 来自 target-dev 困难切片，该结果用于验证方法机制和暗光专项有效性，不是完全独立的跨序列 zero-shot 泛化声明。完整 real 域的剩余 MCML 主要由远距离候选不足和小目标几何误差造成，属于后续独立研究问题。

**创新点定义**：

> 针对暗光大目标场景中几何候选仍存在但分类排序坍塌的问题，本文提出一种 scope-gated frozen-DINO rotated rescue detector。该方法冻结 DINOv2 ViT-L/14，仅使用 source 标注训练旋转 RPN/ROI 小头，并在低照度模式内以 DINO 语义路径替代失效的 anchor/FPN 分类路由；随后采用完全由 source validation 选择的因果旋转框稳定器，在不改变检测中心、分数、排序和输出存在性的前提下平滑尺度与周期角度。完整顺序推理中 scope 外严格保留 BrightAug，因而实现暗光大目标排序恢复、时序稳定性恢复和非目标域能力隔离。

**当前还不能宣称**：已经解决完整 real test 的所有 MCML；已经解决远距离/小目标候选生成；已经完成不依赖 target-derived scope 的最终部署方法。

<a id="sec-3-2"></a>

### 3.2 远距离与小目标：从候选覆盖到最终排序

<a id="sec-3-2-1"></a>

#### 3.2.1 问题定位

暗光 DINO 分支的检测头只有一个 stride-14 patch-grid 特征层：

```text
Frozen DINOv2 ViT-L/14
  -> final patch tokens, stride 14
  -> single-scale OrientedRPNHead
  -> RotatedRoIAlign, featmap_strides=[14]
  -> RotatedShared2FCBBoxHead
```

因此小目标阶段的核心问题不是「DINO 是否有语义」，而是这些语义能否以足够密集的空间分辨率形成候选，并在 ROI/NMS 阶段保留正确顺序。三个困难切片必须分开处理：`real_seq02[137..169]` 暗光大目标是分类排序坍塌（已在 3.1 解决）；`real_seq02[2..41]` 是远距离目标；`real_seq03[129..192]` 是短边约 `1.124` 个 DINO token 的更小旋转目标。

<a id="sec-3-2-2"></a>

#### 3.2.2 候选覆盖审计与提升

**远距候选初筛（已被取代）**：`crane_project/tools/dino_teacher_far_distance_candidate_audit.py` 检查原冻结 DINO labeller 的最终输出能否覆盖 `real_seq02[2..41]`，早期日志为 `geometry=0/40`、`R@20=0`、`R@100=0`、`top1=0`、`decision=DINO_FAR_DISTANCE_CANDIDATE_GENERATION_INSUFFICIENT`。该数字只反映当时特定候选输出路径的最终可用结果，不等价于逐级 RPN proposal 覆盖。后续 RPN-to-ROI attrition 审计证明正式 DINO 的 `seq02_far` RPN 实际为 `40/40`，因此**不能继续引用 `0/40`** 作为「DINO RPN 完全没有远距候选」的结论。该工具由更细的逐级审计取代，仅保留为实验历史。

**Token-Scale / RPN Coverage Audit**（`dino_teacher_token_scale_rpn_coverage_audit.py`，输出 `token_scale_rpn_coverage_audit_v1.json` 等）区分三种解释：目标比 DINO patch 还小、anchor 本身无几何覆盖、anchor 可覆盖但训练后的 RPN 未把正确 proposal 排进候选池。审计严格只读：`optimizer_steps=0`、`checkpoint_writes=0`、DINO/RPN/ROI 参数均未变化。

| 数据 | 短边 token 中位数 | anchor@0.5 | RPN@20 | RPN@2000 |
| --- | ---: | ---: | ---: | ---: |
| source-small | — | 约 0.969 | 0.589 | 0.726 |
| `seq02_far` | 1.498 | 1.000 | 0.900 | 1.000 |
| `seq02_dark` | 2.683 | 1.000 | 0.788 | 1.000 |
| `seq03_small` | 1.124 | 0.984 | 0.594 | 0.703 |

三个 checkpoint（正式 DINO、small-hard ROI 分类头、source-safe 插值头）的 RPN 数字相同，说明它们都没有修改 RPN；这是正确现象，不是缓存错误。结论：`seq03_small` 的理论 anchor 基本存在（anchor 覆盖 `63/64`、assignment `64/64`），真正不足的是 stride-14 DINO 特征上已学习的 RPN objectness/regression；单纯继续训练 ROI 分类器不能提高 RPN@K。

**RPN-to-ROI Attrition + Latency Audit**（`dino_teacher_rpn_roi_attrition_latency_audit.py`）逐级定位候选从 RPN 到最终 top-1 的损失位置：RPN proposal → ROI regression → decoded geometry → rotated NMS → valid-content filter → final top-1 ordering。V1 存在归因口径不完整的问题，V2 增加了 NMS/ordering/regression 的互斥终端原因和一致计数，后续以 V2 为准。使用 `source_safe_interpolated_head.pth` 的 V2 结果：

| 切片 | RPN recall | decoded geometry | final top1 | MCML | 终端失败 |
| --- | ---: | ---: | ---: | ---: | --- |
| `seq02_far` | 1.000 | 1.000 | 38/40 | 1 | 2 个 ordering/NMS |
| `seq02_dark` | 1.000 | 1.000 | 29/33 | 1 | 4 个 ordering/NMS |
| `seq03_small` | 0.703 | 0.984 | 50/64 | 6 | 13 个 ordering/NMS，1 个 regression |

`seq03_small` 的 decoded geometry 高于 RPN recall 并不矛盾：部分初始 proposal 的 RIoU 小于 0.5，但 ROI regression 能把它们修正为可用旋转框。可靠结论是 `seq03_small` 不是单一 RPN miss，而是 stride-14 候选覆盖不足 + ROI ordering/NMS 的复合问题，且 regression 只造成 1/14 个终端失败，不是第一优先级。

延迟与参数量基线（三张 GTX 1080 8G，DINO blocks 分片运行）：

```text
Frozen DINOv2：304,368,640 parameters
RPN + ROI heads：54,824,560 parameters
source DINO branch p50：约 1.740 s/frame
seq03_small p50：约 1.975 s/frame
```

主要耗时来自冻结 DINOv2 前向，而不是 RPN/NMS。

**插值式多尺度 RPN Coverage Audit**（config `crane_symeood_scoped_dino_lowlight_multiscale_v1.py`，`feature_strides=[7,14,28]`）：不训练新参数，只把同一个 stride-14 DINO patch grid 双线性插值/池化为 stride 7/14/28，检查多尺度候选覆盖上界。`optimizer_steps=0`、`audit_trainable_parameters=0`、使用原正式 DINO head 权重。

| 数据 | 单尺度 RPN@20 | 多尺度 RPN@20 | 单尺度 RPN@2000 | 多尺度 RPN@2000 |
| --- | ---: | ---: | ---: | ---: |
| source | 0.713 | 0.710 | 0.835 | 0.874 |
| `seq02_far` | 0.900 | 0.800 | 1.000 | 0.950 |
| `seq02_dark` | 0.788 | 0.636 | 1.000 | 0.909 |
| `seq03_small` | 0.594 | **0.719** | 0.703 | **0.875** |

`seq03_small` RPN@2000 从 `45/64` 提高到 `56/64`，证明提高空间分辨率是有效方向；但朴素共享权重的三尺度同时损伤 far 和 dark，说明不同尺度不能只靠固定插值和同一 RPN 权重机械处理。该 config 仍是实验配置，不能作为正式模型。

**LiDeRe-inspired S7 dense readout（第一版）**：在冻结 DINOv2 S14 特征上增加轻量高分辨率补充分支。

```text
Frozen DINOv2 S14 feature
  |-- frozen native S14 RPN
  `-- 1x1 1024->128 -> 2x bilinear upsample
       -> depthwise 3x3 + GroupNorm + GELU + pointwise 1x1
       -> zero-initialized residual gate
       -> trainable S7 Oriented RPN
native S14 proposals + bounded S7 proposals -> one shared native ROI head
```

冻结 DINOv2、原 S14 RPN、ROI classifier `alpha=0.5`、ROI bbox regression；只训练 S7 readout + S7 RPN（`309,994` 参数）；target 未读取。source baseline 为 `source_safe_interpolated_head.pth`（full `677/738`、small `303/350`、MCML `3`、旧正确帧 `677`）。

| epoch | mean loss | full top1 / MCML | small top1 / MCML | 保留旧正确帧 | gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | — | 677 / 3 | 303 / 3 | 677/677 | PASS，回退基线 |
| 1 | 0.14666 | 660 / 11 | 287 / 11 | 651/677 | FAIL |
| 2 | 0.02276 | **688 / 3** | **318 / 3** | 654/677 | FAIL |
| 3 | 0.01965 | 684 / 2 | 312 / 2 | 653/677 | FAIL |
| 4 | 0.01942 | 685 / 2 | 312 / 2 | 655/677 | FAIL |

其他记录：`S7 trainable parameters = 309,994`、`DINO parameters unchanged = True`、head peak allocated memory 约 `346 MB`、all epoch feature-cache hits `2781/2781`、`target_used_for_training = False`、`target_used_for_checkpoint_selection = False`。

解释：epoch 2 的小目标 source top-1 从 `303/350` 提高到 `318/350`，说明 S7 readout/RPN 确实学到了额外候选；但同时丢失 23 个原本正确的 source 帧——新增 S7 候选进入同一个 ROI/NMS 排序后抢走了部分 native S14 正确候选。这是**方向上的正证据、部署门控上的失败**，不是训练崩溃。裁决：`S7 first implementation = source candidate gain, exact-retention failure`、`best_epoch = 0`、selected checkpoint 为 S7 disabled 的 native `alpha=0.5` 行为、target 未运行且未授权。

source-only 选择门槛为：从 source-safe `alpha=0.5` 初始 checkpoint 起旧正确帧保留率 `100%`、source full top1 `>=677/738`、source-small top1 `>=303/350`、source MCML `<=3`，且候选 epoch 的 source-small 选择键必须优于初始 checkpoint。选择键按 `top1 -> R@20 -> R@100 -> mean RIoU` 的顺序比较，因此允许 top-1 持平但 proposal 排名改善；`RPN@2000` 是 checkpoint 固定后的归因指标，不作为训练脚本的直接选择门槛。这些门槛使用 source-val；target 不用于选择 readout 结构、epoch、阈值或 checkpoint。该模块没有实现 LiDeRe 的完整 readout/attention 结构，只借鉴「冻结大 backbone + 轻量 dense readout」原则，因此不能称为 LiDeRe 复现，只能报告为第一版消融和失败原因。

**高分辨率 S7 的候选覆盖结论**：protocol-25 的 source-safe high-resolution ROI readout 在 source 达到 full/small `688/738`、`311/350`、`lost=0`；固定 target-dev 中 `seq03_small` R@20 `55/64→63/64`、R@100 `55/64→64/64`。这与「远距候选初筛 `0/40`」形成完整修正链：远端候选实际存在，小目标候选覆盖可通过提高读出分辨率补足。

<a id="sec-3-2-3"></a>

#### 3.2.3 安全基线的确定

**Source-Safe FC Classifier Weight Interpolation**（`crane_project/tools/dino_teacher_fc_cls_interpolation_selector.py`，输出 `source_interpolation_result.json`、`work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth`）在正式分类器与更新后分类器之间插值，只使用 source-val 选择最大安全更新幅度：

```text
W(alpha) = (1-alpha) * W_formal + alpha * W_updated
```

这是一项**分类器权重插值**，不是 DINO 空间特征上采样。

| alpha | full top1 / MCML | small top1 / MCML | 旧正确帧丢失 | Gate |
| ---: | ---: | ---: | ---: | --- |
| 0.000 | 662 / 7 | 293 / 7 | 0 | baseline |
| 0.125 | 666 / 5 | 297 / 5 | 0 | PASS |
| 0.250 | 671 / 4 | 300 / 4 | 0 | PASS |
| **0.500** | **677 / 3** | **303 / 3** | **0** | **PASS / selected** |
| 0.750 | 678 / 3 | 303 / 3 | 3 | FAIL |
| 1.000 | 679 / 4 | 304 / 4 | 10 | FAIL |

`alpha=0.5` 完整保留原先 662 个正确 source 帧并新增 15 个正确帧，是当前最强且满足 exact retention 的分类器 checkpoint。该实验已完成且有效，**不要重复权重插值**。

**Rotated-NMS Retention Audit**（`dino_teacher_rotated_nms_retention_audit.py`）判断原 DINO ROI `nms_iou_thr=0.1` 是否过强抑制小目标附近的正确框，只用 source-val 选择 NMS 策略：

```text
selected NMS IoU = 0.5
source top1 = 662/738
source exact retention = 662/662
source-small post-NMS R@20 = 0.920
```

target diagnosis：`seq02_far` `38/40`、MCML `1`、post-NMS R@20 `0.975`；`seq02_dark` `29/33`、MCML `1`、`0.970`；`seq03_small` `52/64`、MCML `6`、`0.875`。`0.5` 是 source-selected 的更合理 DINO ROI NMS 阈值，能保留更多正确 ROI，但它仍是推理策略，不是空间表征创新，也没有让 MCML 从 6 降下来。该 NMS 只属于 DINO ROI head，不改变原 `crane_symeood_k1.py` / BrightAug 的无 NMS 设计，因此不存在结构冲突。保留为后续统一 DINO 的固定推理设置和候选接口，不单独作为论文创新。

<a id="sec-3-2-4"></a>

#### 3.2.4 最终排序与候选接管实验族

以下 27 条记录按协议演进顺序排列。它们方法互不相同，但共同回答同一个问题：**在不破坏 native 旧正确帧的前提下，能否让更好的 S7 候选接管 native top-1。** 每条记录保留方法差异、冻结条件、关键数据、失败原因和停止条件。

**（1）ROI Classification Small-Hard 训练**（`dino_teacher_roi_cls_small_hard_v1`，4 epochs，仅训练 `fc_cls.weight`/`fc_cls.bias` 共 2050 参数）

目标：不修改 DINO、RPN、ROI regression，用 source-small hard samples 改善 ROI 分类排序。

| 模型 | full top1 / MCML | small top1 / MCML |
| --- | ---: | ---: |
| 正式 DINO | 662/738 / 7 | 293/350 / 7 |
| small-hard best | 679/738 / 4 | 304/350 / 4 |

target V2 attrition：`seq02_far` `39/40`；`seq02_dark` `29/33`；`seq03_small` `47/64`。结论：source 指标明显改善，但直接使用完整 small-hard 分类器没有在 `seq03_small` 上稳定迁移；只看 source top1 会掩盖旧正确帧被替换和 target 排序变化。裁决：不直接部署，但保留为可插值的新分类器端点（后由第 3.2.3 节插值选出）。

**（2）Pairwise Ranking + Retention V1**（`dino_teacher_roi_cls_pairwise_retain_v1`，仅训练 ROI `fc_cls` 2050 参数 ）

目标：对每个 source 正样本与 hard negative 建立 pairwise margin，同时用原分类器 logits 作为 retention teacher，直接学习「正确 ROI 应排在相似错误 ROI 前面」。4 个 epoch 都改善 source top1，但没有满足 exact retention，正式 selector 回退到 epoch 0：

| epoch | full top1 / MCML | small top1 / MCML | exact retention |
| ---: | ---: | ---: | --- |
| 1 | 678 / 4 | 302 / 4 | FAIL |
| 2 | 677 / 4 | 306 / 4 | FAIL |
| 3 | 679 / 4 | 307 / 4 | FAIL |
| 4 | 677 / 4 | 306 / 4 | FAIL |

随后的 source-only epoch 重选允许最低保留率 `0.985`，选出 epoch 1：`retention = 0.986405`、full `678/738`、small `302/350`、checkpoint `labeller_best_source_retained_985.pth`。三切片 attrition 为 `seq02_far 39/40 MCML=1`、`seq02_dark 29/33 MCML=1`、`seq03_small 45/64 MCML=6`。结论：对 far slice 有利，但没有解决 `seq03_small`，且 exact retention 未通过。裁决：不作为最终 checkpoint，分类方向只用于后续安全权重插值。

**（3）Pairwise Ranking V2**（`dino_teacher_roi_cls_pairwise_v2_formal4_v1`，4 epochs）

目标：只在 source 中确实发生 top-1/NMS 排序失败的帧上构造 paired positive/negative，并要求 exact source retention。

| epoch | full top1 / MCML | small top1 / MCML | 旧正确帧丢失 | pair signal |
| ---: | ---: | ---: | ---: | --- |
| 1 | 674 / 3 | 301 / 3 | 3 | 极少 |
| 2 | 672 / 5 | 299 / 5 | 2 | 极少 |
| 3 | 670 / 5 | 299 / 5 | 2 | 0 |
| 4 | 670 / 5 | 299 / 5 | 2 | 0 |

selector 因 exact retention 失败回退 epoch 0，且被旧 `alpha=0.5` 安全插值严格支配（安全插值 `677/738、303/350、MCML=3、丢失 0` vs V2 epoch1 `674/738、301/350、MCML=3、丢失 3`）。裁决：关闭，不要对 V2 再做一轮分类权重插值。

**（4）retention-aware 全局 affine merge**（`--train-components s7_merge`；config `crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py`）

方法：只训练 `s7_score_calibrator.raw_scale` / `bias` 两个标量，对 S7 lane 做全局 affine 校准；候选来源标记 `native_s14` / `supplement_s7`；合并位置在 shared ROI 解码后、最终排序前；两个来源分别执行 NMS，S7 不删除 native 候选。冻结：DINOv2、原 S14 RPN、ROI classifier `alpha=0.5`、ROI bbox regression。target 在 source gate 通过前不读取。

| epoch | full top1 / MCML | small top1 / MCML | 保留旧正确帧 | S7 top1 帧 | gate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 677 / 3 | 303 / 3 | 677/677 | 0 | PASS，回退基线 |
| 1 | 688 / 3 | 311 / 3 | 676/677 | 42 | FAIL |
| 2 | 687 / 3 | 311 / 3 | 675/677 | 54 | FAIL |
| 3 | 686 / 3 | 310 / 3 | 674/677 | 55 | FAIL |
| 4 | 686 / 3 | 310 / 3 | 674/677 | 55 | FAIL |

候选覆盖明显改善（full `R@100 717 -> 738`、source-small `329 -> 350`），但训练集中约 `2774/2781` 个 retention pair 的 hinge 从第一轮开始就是 0，实际梯度主要来自约 7 个 gain pair；全局 affine 因此持续抬高 S7 分数，增益帧固定为 12，丢失帧却从 1 增加到 3。该结果拒绝的是**两标量全局校准**，不是否定 S7 候选或 LiDeRe-inspired readout。

配套 source-only 冲突审计（`--source-conflict-result-json` / `--source-conflict-epoch`，只复核 epoch 1 变化的 13 帧）输出 `[source-val] 13/13 top1_hits=12`、`decision=SOURCE_ONLY_CONFLICT_AUDIT_COMPLETE_TARGET_NOT_READ`、`mode=eval-only, no retraining`、`target_dev=null`。`12/13` 与全量 source-val 的 `676/677` retention、`+12/-1` 完全一致。诊断代码同时把每个 epoch 的 affine scale/bias、retention/gain active constraint 数量和 lost/gained 帧的分路 top-1 score、RIoU、来源、pre-NMS log-odds 写入 `source.history[].s7_merge_conflicts`。

随后采用独立的宽松 target-dev 诊断门（`retained>=676/677`、`full top1>=685/738`、`small top1>=308/350`、`full/small MCML<=3`、`gained/lost>=5`），epoch 1 满足该门，仅被授权作为一次 target-dev 诊断候选，不具备部署资格。工具 `crane_project/tools/dino_teacher_s7_relaxed_target_audit.py` 在同一份冻结 DINO 特征上对比 native `alpha=0.5` 与固定 epoch 1，一次运行固定三切片（`work_dirs/dino_teacher_s7_relaxed_target_audit_v1/result.json`）：

```text
decision = S7_RELAXED_TARGET_DEV_DIAGNOSTIC_PASS
eligible_for_deployment = false
eligible_for_final_test = false
eligible_for_next_stage = true
optimizer_steps = 0
```

| 固定切片 | native top-1 | S7 merge top-1 | MCML | usable geometry | R@20 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `seq02_far` | 38/40 | 38/40 | 1 -> 1 | 无变化 | 40 -> 40 | 40 -> 40 |
| `seq02_dark` | 29/33 | 29/33 | 1 -> 1 | 无变化 | 33 -> 33 | 33 -> 33 |
| `seq03_small` | 50/64 | 51/64 | 6 -> 6 | 55 -> 64 | 55 -> 63 | 55 -> 64 |

`seq03_small` top-1 只增加 `1/64`（+1.56 个百分点），但 usable geometry 增加 9 帧、R@20 增加 8 帧、R@100 增加 9 帧，平均 top-1 RIoU 约从 `0.5682` 增至 `0.5789`。逐帧结果显示 S7 分路 top candidate 在 `seq03_small` 有 7 帧成为最终 top-1 且全部正确，其中只有 `real_seq03[176]` 是新 top-1 gain；另有 8 帧的 S7 分路第一候选本身正确却被错误 native 候选压在后面。`seq02_far`、`seq02_dark` 中 S7 没有抢占任何 top-1。该结果证明全局两标量 affine 校准无法表达「同一帧内 native 与 supplement 的条件竞争」，剩余瓶颈转移到 ROI 排序和 native/S7 分路仲裁。`seq03_small` 的 raw detection 数量由 `15697` 增至 `29890`，后续 source gate 通过后还需做独立的延迟/显存检查。

**（5）source-only lane arbitration v1**（`--train-components s7_lane_arbitration`；config `crane_symeood_scoped_dino_lowlight_s7_lane_arbitration_v1.py`；`protocol_version=12`）

方法：加载并冻结 epoch 1 的完整 S7 merge checkpoint，只训练零初始化、输出幅度受限的 `S7LaneArbitrator`（33,057 参数），仅对 supplement S7 的 pre-NMS logit 加残差；输入为 ROI embedding、S7 原始 logit、native top logit。native 分路分数、DINO、native S14 RPN、S7 readout/RPN、ROI 回归和全局 affine 均不更新。损失仍只由 source GT 无梯度挖掘 retention/gain pair，NMS 和 Recall 不进入损失。

| epoch | full top1 / MCML | small top1 / MCML | retained / lost / gained | S7 top1 | gate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 677 / 3 | 303 / 3 | 677 / 0 / 0 | 0 | PASS，正式回退 |
| 1 | 690 / 3 | 316 / 3 | 672 / 5 / 18 | 107 | FAIL |
| 2 | 688 / 3 | 313 / 3 | 673 / 4 / 15 | 79 | FAIL |
| 3 | 688 / 3 | 315 / 3 | 670 / 7 / 18 | 105 | FAIL |
| 4 | 687 / 3 | 315 / 3 | 669 / 8 / 18 | 108 | FAIL |

总体指标比全局 affine 更高，但所有训练 epoch 都破坏旧正确帧。四个 source 帧在全部 epoch 持续丢失：`real_seq07/215`、`real_seq07/220`、`sim_seq10/116`、`sim_seq10/20`。正式 `best_epoch=0`，selected checkpoint 是回退行为；不能拿 epoch 1 当部署模型，也没有授权 target 或完整 test。

**（6）lane arbitration v2 + gain replay**（`protocol_version=13`）

方法变化：使用动态 current-adjusted top-4 hard negatives，并把 7 个 source gain 帧重复 8 次形成 49 条额外 replay 记录。权威副本必须以 `protocol_version=13` 且 `source_selected_checkpoint` 指向 `dino_teacher_s7_lane_arbitration_v2` 为准；此前同名副本中若只把该路径写成 `_v1`，数值内容相同但元数据路径错误，不作为权威副本。

| epoch | full top1 / MCML | small top1 / MCML | retained / lost / gained | S7 top1 | val lane mean / max | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 677 / 3 | 303 / 3 | 677 / 0 / 0 | 0 | 0 / 0 | PASS，正式回退 |
| 1 | 681 / 3 | 314 / 3 | 647 / 30 / 34 | 400 | 1.9939 / 2.0 | FAIL |
| 2 | 683 / 3 | 315 / 3 | 650 / 27 / 33 | 391 | 1.9882 / 2.0 | FAIL |
| 3 | 683 / 3 | 315 / 3 | 650 / 27 / 33 | 390 | 1.9816 / 2.0 | FAIL |
| 4 | 683 / 3 | 315 / 3 | 650 / 27 / 33 | 390 | 1.9806 / 2.0 | FAIL |

v2 没有学到条件化仲裁，而是接近对每帧全部 S7 候选统一加最大允许值 `+2`：训练侧 mean adjustment 从 epoch 1 的 `0.0586` 升到 epoch 3/4 的约 `1.997`，source-val 上约 390–400/738 帧由 S7 成为 top-1，prior loss 增至约 `0.03987` 仍无法阻止 tanh 饱和。gain replay 放大了极少数正向样本，却把错误 S7 候选一起抬高，结果明显劣于 v1。isolation 干净：DINO 和所有冻结 head 参数未改变，target 未用于训练、选择或评估。失败属于目标函数/参数化问题。正式 `best_epoch=0`，不进行 target 或完整 test。

**（7）non-positive S7 quality suppression v1**（`--train-components s7_quality_suppression`；config `crane_symeood_scoped_dino_lowlight_s7_quality_suppression_v1.py`；`protocol_version=14`）

方法：固定 affine epoch 1 作为起点，只在 source train 学习 S7 lane 的可用性/质量，并施加同一帧共享的非正值惩罚。实现为 source-only 风险预测器，根据 ROI embedding、raw/affine log-odds、native top log-odds 和 lane gap 预测风险：

```text
r = source-only S7 lane risk logit
strength = clamp(ReLU(r), 0, 1)
delta = -D * strength，D=2，delta in [-2, 0]
z_s7_new = z_s7_affine + delta
```

输出层零初始化，因此训练开始时 `delta=0` 精确复现 affine epoch 1；风险 BCE 在边界仍有梯度。错误 S7 只有在 native top 正确且距抢占 margin 小于 `0.5` 时才成为 risk pair；S7 top 本身 `RIoU>=0.5` 时只约束 `delta` 接近 0，不增加任何正值。惩罚对该帧所有 S7 logits 一致，保持 S7 内部排序；由于 `delta<=0`，不能相对旧 affine 基线制造新的 S7 overtakes。正式门槛为 `lost=0`、`full>=688`、`small>=311`、`full/small MCML<=3`，并取消 `lost<=1` 的宽松诊断门。

结果：唯一可训练模块为 `s7_quality_suppressor.*`，每帧一个共享 `delta in [-2,0]`；禁止正向 promotion 与 gain replay；target 未读取。权威训练结果为 source train risk pairs `0/2781`，四个 epoch 的 `delta` 全为 0；affine 候选结果 `full 688/738、small 311/350、retained 676/677`；official `best_epoch=0`、S7 disabled。原因分析：2781 个 source-train 帧中有 2755 个 preserve pair、26 个 S7 top-1 错误帧，但没有一帧同时满足 `native top correct + S7 top wrong + within margin`，因此 risk/retention loss 始终为 0，只有 preserve BCE 下降——这不是有效 suppression 学习，而是风险标签无正样本造成的 no-op。训练起点虽然是 affine epoch 1，但全部 epoch 失败时正式 best checkpoint 回退到 S7-disabled 的 native `alpha=0.5`，而不是回退到不安全的 affine checkpoint。

2026-08-02 把协议升级为 v15，增加训练前 source-only support audit：预检复用正式训练的同一前向和同一风险判定，报告全部 S7 错误帧的 native/S7 RIoU、affine gap、margin 排除原因和序列分布；若 `risk_pair_count < 1`，程序在 epoch 1 前停止，保留 epoch 0，写出 `SOURCE_ONLY_TRAINING_SKIPPED_ZERO_S7_QUALITY_RISK_SUPPORT`，不再产生伪收敛训练。v15 预检只用于固化失败证据和防止误训，不构成新的模型实验。

**（8）统一跨尺度因果时序候选选择 v1**（`--s7-temporal-association`；`crane_project/utils/s7_temporal_association.py`；`protocol_version=16`）

设计原则：不再按 `seq02_dark` / `seq03_small` 或任何预定义 target 片段路由，而是在所有连续视频上统一运行跨尺度候选选择。第一版不是新的 candidate quality head，而是在固定 affine epoch 1 的 native/S7 post-NMS top-100 候选上，只拟合六个非负多线索权重：

```text
calibrated score logit
normalized center distance
rotated IoU
log-scale consistency
pi-periodic angle consistency
DINO ROI appearance cosine similarity
```

训练记录按 `split/seq/frame` 排序，只在帧号连续时使用上一帧 source GT usable candidate 构造关联监督；推理只使用上一帧已选择候选。首帧、序列切换、帧号缺口或连续性失败均精确回退 native top-1；非 native 候选默认需要连续两帧确认后才能接管。source-small 指标从完整连续 source-val pass 中切片统计，不再对 small 子集单独运行。部署配置显式使用 `scope_policy=all_frames`、`scope_manifest=None`，不读取 dark/small scope 或 sequence id 作为模型特征。第一轮执行正式 gate（`lost=0`、full `>=688/738`、small `>=311/350`、full/small `MCML<=3`）并要求完整 source 连续流的 DFR 不高于 native、ACI 不低于 native（沿用项目 evaluator 的 `35°` ACI 上限）。

结果（外部迁移文件 `/Users/mac/Downloads/Copy of train_result.json`，`protocol_version=16`）：四个 epoch 的离散输出完全相同，full `680/738`、small `305/350`、MCML 均为 `3`，exact retention `677/677`、`lost=0`、`gained=3`。时序关联消除了 affine merge 的唯一 source 丢失，但没有达到正式 `688/311` 门槛，正式 `best_epoch=0`，target 未读取。时序指标 DFR 从 `0.08464` 降至 `0.08272`，ACI 从 `0.82687` 升至 `0.83092`；候选 R@100 达到 full `738/738`、small `350/350`，说明主要短板是选择增益而不是候选覆盖。source train 中只有 `7/2781` 个 gain pair，无法为六个全局 cue 权重提供足够稠密的增益监督。该轮证明 native-safe 时序选择可实现 `lost=0`，但不能取代正式基线。

随后增加三项不改变模型目标的安全诊断：

```text
1. temporal_association.source_selected=True 时，运行时强制检查 checkpoint best_epoch、
   source gate、exact retention 和 full/small/MCML；失败即拒绝启用 S7，防止 epoch-0
   fallback checkpoint 被时序配置误部署。
2. 训练前记录初始六权重的 source temporal summary，区分初始化收益与训练收益。
3. 每个 source temporal summary 记录 candidate margin、continuity、override、pending
   confirmation、reset 等阻断原因；不使用 target、不改变 checkpoint 选择。
```

相关代码入口：`source_selected_checkpoint_gate`、`summarize_temporal_association_audit`、`CausalTemporalCandidateSelector` 的诊断字段。本地回归结果：相关测试 `114 passed`，`py_compile` 与 `git diff --check` 均通过。

**（9）候选级连续 RIoU quality head**（`--s7-temporal-quality-head`；`crane_project/utils/s7_temporal_association.py`）

由于 source train 的 gain pair 稀疏源于视频分布差异，本阶段不再依赖 gain-pair miner、gain replay 或正向 lane promotion。新增 `S7CandidateQualityHead`，对固定 affine epoch-1 的 native/S7 post-NMS top-100 候选逐候选计算 source GT max-RIoU，并使用 `weighted_smooth_l1(sigmoid(logit), target)`（权重 `1+3*target`）做 dense supervision。quality logit 只作为第七个 causal temporal cue；六个原始 cue 权重、DINO、native/S7 proposal、ROI head 和 affine calibrator 全部冻结。quality head 输出层零初始化，训练前的候选排序与已审计时序基线完全一致。部署和训练均保持统一全序列策略（`scope_policy=all_frames`、`scope_manifest=None`），native fallback、两帧确认、sequence/frame-gap reset、DFR/ACI 与 exact-retention gate 不变。

结果：四个 epoch 均为 full `681/738`、small `306/350`、`lost=0`，正式 `best_epoch=0`。

**（10）固定 epoch-4 source-only 时序归因审计**（协议升级为 `protocol_version=19`；入口组合 `--source-temporal-attribution-audit --source-temporal-attribution-epoch 4 --eval-only-checkpoint <labeller_epoch_04_source_only.pth> --skip-target-eval`）

目的：在不训练、不更新参数、不重新选择 best epoch、不读取 target 的前提下，判断上面 `+7 full / +5 small` 的差距来自第七个 quality cue、七 cue 融合、margin/continuity，还是两帧 confirmation。每帧在不改变 selector 输出和时序状态的前提下记录：native fallback、quality-only 排序及正确候选 rank、七 cue fused argmax、margin 反事实、margin+continuity 的 pre-confirmation 反事实、pending confirmation 候选、最终 selected candidate。输出 `source_temporal_attribution_audit` 汇总 full 与 source-small 的 top-1、相对 fallback 的 gain/loss、quality rank、pending correctness。只有 pre-confirmation 同时达到 full `>=688`、small `>=311` 且相对 fallback `lost=0`，才输出 `ALLOW_ONE_BOUNDED_CONFIRMATION_RULE_REVISION`；否则输出 `CLOSE_CURRENT_QUALITY_TEMPORAL_MERGE_KEEP_NATIVE_BASELINE`。pre-confirmation 是固定当前时序状态的一步反事实，不会递归更新后续时序状态，因此即使通过也只授权一次受限 confirmation 规则修改。实现入口：`temporal_selection_attribution`、`summarize_temporal_readonly_attribution`、`build_source_temporal_attribution_audit`；本地检查 `123 passed`。

**（11）递归 immediate-override 审计与阶段二B relative-quality**（`--source-temporal-immediate-override-audit`；`protocol_version=16`→阶段二B）

考虑后半部分将增加自适应协方差 EKF/卡尔曼后处理，时序候选不再把「两帧确认」视为最终不可放宽的约束，但仍保留 native fallback、candidate top-100、margin、continuity、sequence/frame-gap reset 和 source-only 隔离，不允许 quality-only 排序或无条件 S7 promotion。该入口固定读取 rejected epoch-4 checkpoint，仅在 `candidate_margin_ok && candidate_continuity_ok` 时把运行时 confirmation 设为 1 帧，并真实递归更新下一帧 temporal state，用于验证此前一步反事实 `690/738、312/350` 能否在真实顺序推理中保持。不训练、不更新参数、不选择 checkpoint、不读取 target，输出写入 `source_temporal_immediate_override_audit`。对应的严格只读工具为 `crane_project/tools/dino_teacher_s7_temporal_immediate_override_audit.py` 与 `crane_project/tools/dino_teacher_s7_temporal_source_attribution_audit.py`，两者都强制 `--eval-only-checkpoint`、`--skip-target-eval` 和相应的 audit 模式开关。

epoch-4 source attribution 已输出 `ALLOW_ONE_BOUNDED_CONFIRMATION_RULE_REVISION`：margin+continuity 的 pre-confirmation 在 full `691/738`、small `312/350` 上相对 native fallback 均 `lost=0`。

阶段二B relative-quality source-only 训练已完成，权威迁移结果为 `/Users/mac/Downloads/Copy of Copy of Copy of Copy of train_result.json`，正式选择 `source.best_epoch=4`：full `691/738`、small `312/350`、full/small MCML 均为 `3`、exact retention `677/677`、`lost=0`、R@100 达到 `738/738` 与 `350/350`，DFR 和 ACI 也通过 source 时序非退化门。训练只更新 candidate quality head，target 未读取。相对 pointwise immediate-override 初始化，离散 top-1 没有继续增加，因此不再增加 epoch 或扫描 relative loss 超参数；本轮结论是「source-safe 排序细化通过」，不是把 relative loss 单独宣称为新的 top-1 增益来源。

**（12）固定三切片 target-dev 入口与阶段三授权门**（严格只读）

`crane_project/tools/dino_teacher_s7_temporal_fixed_target_audit.py` 只接受当前 source-selected epoch-4 relative-quality checkpoint，要求 `lost=0`、full `>=688`、small `>=311`、MCML `<=3`、DFR/ACI 非退化，并验证 checkpoint 内部 source gate、relative-quality、quality head、`min_confirmations=1` 和 S7 inference 元数据；候选 checkpoint 必须与训练结果中的 `source_selected_checkpoint` 路径一致。target 切片、阈值和门槛全部在代码内固定，命令行不能修改：

```text
seq02_far:   test/real_seq02[2..41]，baseline 38/40，MCML=1
seq02_dark:  test/real_seq02[137..169]，baseline 29/33，MCML=1
seq03_small: test/real_seq03[129..192]，baseline 50/64，MCML=6，R@100=55/64
```

阶段三授权门：far/dark top-1 与 MCML 不回退并达到上述绝对参考线；`seq03_small` 至少 `51/64`、MCML `<=6`、R@100 `64/64`。DFR/ACI 继续完整报告，但考虑后续独立 EKF/卡尔曼阶段，本次不作为 target-dev 硬门。运行过程 optimizer steps 为 0，参数版本必须保持不变；输出只决定是否授权阶段三学生训练，不构成部署或完整 test 证据。

实际结果为 `seq02_far=39/40`、`seq02_dark=29/33`、`seq03_small=50/64`；后者 R@100 已达 `64/64`，但没有达到预注册的严格 Top-1 `51/64`，因此阶段三仍未授权。该差距不能事后放宽为通过。配套的 `crane_project/tools/dino_teacher_s7_temporal_source_attribution_audit.py` 在 source full / short-token small 两组上记录 native fallback、quality-only、七 cue fused、margin、pre-confirmation 和 final selection 的逐帧归因；只有 source-only pre-confirmation 同时达到 full `>=688`、small `>=311` 且相对 fallback `lost=0`，才允许把结果解释为一次受限 confirmation-rule 新实验的依据。

**（13）阶段三 source-only 候选排序 student**（`crane_project/tools/dino_teacher_s7_temporal_student_train.py`；config `crane_symeood_scoped_dino_lowlight_s7_temporal_student_v1.py`）

数据边界：当前数据只有 `train/train_sim/val/test`，没有与 test 隔离的无标签 target-train split，本阶段明确禁止读取 test 伪标签，实现的是 source-only 候选排序学生，而不是 target pseudo-label student。

固定 teacher 为阶段二 source-selected epoch 4：DINOv2、native S14 RPN、S7 readout/RPN、ROI classifier/regressor、global affine calibrator、七 cue scorer 和阶段二 candidate-quality head 全部冻结。新增独立 `s7_candidate_student_head`，先精确复制 teacher，再仅训练该学生头：

```text
L_stage3 = L_continuous_source_RIoU
         + 0.5 * L_same_frame_relative_quality
         + 1.0 * L_teacher_Bernoulli_distillation
```

短边不超过 4 个 DINO token 的 source GT 帧只在监督损失上乘 `2.0`；该尺度标签不是推理输入，不读取序列名，也不对 `seq03_small` 或三段 target-dev 做路由。推理仍为同一 native/S7 top-100 候选池、固定七 cue、margin+continuity、one-frame confirmation、native fallback 和 causal reset。安全约束：初始化 checkpoint 必须等于阶段二结果中的 `source_selected_checkpoint` 且通过 `best_epoch=4`、source gate、exact retention；复制后先重新验证阶段二 full/small/DFR/ACI，不一致即停止；epoch 1–4 只有同时满足 `lost=0`、full `>=688`、small `>=311`、full/small `MCML<=3`、DFR/ACI 不退化，并且 selection key 严格优于复制 teacher，才可替代 epoch-0 阶段二 fallback。只有 `best_epoch>0` 才授权下一次固定三切片 target-dev；不得直接运行完整 test。

阶段三 fixed target-dev 结果（`/Users/mac/Downloads/target_dev_result.json`，候选 `dino_teacher_s7_temporal_student_v1/labeller_best_source_only.pth`，`best_epoch=4`，`optimizer_steps=0`，DINO/baseline head/candidate head 参数均未更新）：

| 切片 | baseline Top-1 | student Top-1 | baseline MCML | student MCML | baseline R@20/R@100 | student R@20/R@100 | target gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `seq02_far` | 38/40 | 39/40 | 1 | 1 | 40/40 | 40/40 | 通过 |
| `seq02_dark` | 29/33 | 29/33 | 1 | 1 | 33/33 | 33/33 | 通过 |
| `seq03_small` | 50/64 | 50/64 | 6 | 6 | 55/55 | 63/64 | 失败 |

`seq03_small` 的 `MCML=6` 满足本切片预注册的 `MCML<=6`，失败原因不是 MCML，而是严格 Top-1 门要求 `51/64`，实际仍为 `50/64`。本轮同时确认候选覆盖已经达到 `R@100=64/64`，剩余 14 帧属于候选存在但最终没有排到第一的问题。

本轮 target-dev evaluator 自定义时序指标（DFR 为百分比形式，ACI 为无量纲）：

| 切片 | 模型 | mean Top-1 RIoU | DFR (%/frame) | ACI | temporal override | reset |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `seq02_far` | baseline | 0.61218 | 6.2213 | 0.91711 | 0 | 0 |
| `seq02_far` | student | 0.61735 | 6.3526 | 0.94417 | 1 | 1 |
| `seq02_dark` | baseline | 0.60913 | 13.2427 | 0.76817 | 0 | 0 |
| `seq02_dark` | student | 0.60913 | 13.2427 | 0.76817 | 0 | 1 |
| `seq03_small` | baseline | 0.56820 | 9.2120 | 0.87517 | 0 | 0 |
| `seq03_small` | student | 0.57436 | 9.4894 | 0.88085 | 4 | 1 |

本轮结果没有导出 `R_center` 和 `TDR_w10`：它们属于完整系统级 evaluator 的指标，不能由当前 target-dev 的 Top-1 RIoU 摘要可靠反推。后续若进行正式 full-test，必须使用统一的 `R_center` 中心误差阈值、`TDR_w10` 窗口定义、silence 处理和 MCML 口径，并同时输出 `R_center/DFR/TDR_w10/ACI/MCML`。

裁决：

```text
decision = S7_TEMPORAL_STUDENT_FIXED_TARGET_DEV_FAIL_KEEP_SOURCE_MODEL
eligible_for_deployment = false
eligible_for_final_test = false
```

阶段三已经证明 source-safe，但没有在 target-dev 上改善最终小目标排序，因此不能授权完整 test。`MCML=6` 可以作为 `seq03_small` 的诊断结果保留，但不能抵消预注册的 `51/64` Top-1 条件，也不能事后放宽 gate。

**（14）source-only 静态域泛化 ranker**（`protocol_version=22`，`/Users/mac/Downloads/Copy of train_result.json`）

方法：固定 DINOv2、native S14/S7 proposal、ROI head、global affine 和既有时序/quality 组件，只训练 `s7_candidate_static_head`（132,609 参数）。训练监督仅来自 `train + train_sim` 的 source GT，并使用通用 brightness/blur/scale feature-domain augmentation 和同帧 hard-negative relative ranking；推理不使用时序、不读取序列名、不按困难切片路由，也不读取 target。

| epoch | full Top-1 | small Top-1 | full/small MCML | full R@20/R@100 | small R@20/R@100 | lost/gained | source gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 691/738 | 315/350 | 3/3 | 736/738 | 350/350 | 2/16 | 失败 |
| 2 | **695/738** | **318/350** | 3/3 | 736/738 | 350/350 | **1/19** | 失败 |
| 3 | 693/738 | 315/350 | 3/3 | 736/738 | 350/350 | 1/17 | 失败 |
| 4 | 694/738 | 316/350 | 3/3 | 736/738 | 350/350 | 1/18 | 失败 |

epoch 2 是 raw 指标最好的训练状态，DFR 从 native 的 `0.08464` 降至 `0.07934`、ACI 从 `0.82687` 升至 `0.83428`，但仍破坏了旧正确帧 `val|real_seq07|215`；其他 epoch 也分别丢失 1–2 个旧正确帧。所有绝对 Top-1、MCML 和候选覆盖条件均已通过，唯一失败项始终是 `exact_old_correct_retention`，因此 `best_epoch=0` 是预注册安全选择器的正确回退，不是 checkpoint 保存或 best-epoch 代码错误。裁决 `SOURCE_ONLY_STATIC_DOMAIN_RANKER_FALLBACK_TARGET_NOT_READ`、`target_dev=null`。

本轮 `target_dev=null`，训练、checkpoint 选择和评估均没有读取 `seq03_small`，因此不能证明 `seq03_small` 已经提升，也不能把 source epoch 2 的 `695/738` 外推成 target 的 `51/64`。按预注册顺序只有 source gate 通过才允许再读取一次固定 target-dev，本轮四个 epoch 均未通过 exact retention，不再运行 `seq03_small` 对照是协议要求，不是漏测。

**（15）native-protected selective promotion V1**（config `crane_symeood_scoped_dino_lowlight_s7_selective_promotion_v1.py`；入口 `crane_project/tools/dino_teacher_s7_selective_promotion_train.py`）

结构：固定 native S14 + S7 top-100 候选池，冻结阶段二 candidate-quality teacher，只训练 `s7_selective_promotion_head.*` 的 native-vs-S7 成对 advantage / uncertainty head；用

```text
LCB = advantage - lambda * uncertainty
```

超过固定 promotion margin 时才允许单个 S7 候选接管，否则精确回退 native top-1。该路线与已失败的 lane-wide boost、统一 `+2` replay、non-positive lane suppression 和无保护 static residual 结构不同。训练只读取 source GT，使用通用尺度/模糊/亮度扰动和按 source sequence 分组的验证；推理不能把 sequence identity 当作模型特征或特殊路由，也不能读取 `seq03_small` 范围、target GT 或 target-derived threshold。选模在原有 exact-retention 与 full/small/MCML/DFR/ACI 门之外，新增按 `split|sequence` 统计的多序列净增益门（至少两个 source validation sequence 各自净提升）。

实现验证（代码审计阶段）：`native_missing branch = patched and unit-tested`；`py_compile` 通过；`pytest = 137 passed in 1.69s`；`git diff --check` 通过。`native_protected_selective_promotion()` 在 native lane 缺失时返回 `selected_index=None`、保持原始候选顺序、`promoted=False` 和 `reason='native_missing'`，不会把第一个 S7 候选误报为合法输出；`LCB == promotion_margin` 使用 `>=`，边界行为已有测试覆盖；训练只更新 `s7_selective_promotion_head.*`。

服务器使用的阶段二输入为 `/home/omnisky/workspace/symEOOD/work_dirs/dino_teacher_s7_temporal_relative_quality_v1/` 下的 `train_result.json` 与 `labeller_epoch_04_source_only.pth`；训练使用新目录 `work_dirs/dino_teacher_s7_selective_promotion_v2/`，结果中 `target_dev=null`、`target_read=false`、`target_used_for_training=false`、`target_used_for_checkpoint_selection=false`。四个 epoch 的 source-only 结果完全相同：

| epoch | full Top-1 | small Top-1 | full/small R@100 | lost | S7 promotion | source gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |
| 2 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |
| 3 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |
| 4 | 677/738 | 303/350 | 738/738、350/350 | 0 | 0 | 失败 |

失败项是 `full_top1_absolute`、`small_top1_absolute` 和 `multi_sequence_net_gain`；`exact_old_correct_retention`、MCML、DFR 和 ACI 安全条件没有失败。该结果不是训练崩溃，而是 LCB 接管门在当前 source gain 正样本极少的监督下没有授权任何 S7 接管，模型等价于「永远保留 native top-1」。它证明了 fallback 的安全性，但没有证明 selective promotion 能改善 `seq03_small`，因此不能误写成 target 泛化结果。裁决：

```text
implementation = complete
server_source_training = complete
source_gate = failed
source_best_epoch = 0
selective_s7_promotion = 0 for epochs 1..4
target_dev = not_authorized
full_test = not_run
formal_deployable_model = native S14 alpha=0.5, S7 disabled
```

**（16）Two-frame selective ranker V2（标量版）**（`--s7-selective-two-frame`；入口 `crane_project/tools/dino_teacher_s7_small_temporal_ranker_train.py`）

V1 失败不是候选消失，而是 source gain 正样本极少，LCB 门最终选择了零接管。V2 改变信息结构：冻结 phase-2 candidate-quality head，从 S7 lane 的 top-20 候选中只选出一个 quality 最优候选，与 native top-1 构成 pair；tiny head 输入 24 个标量、隐藏层维数固定 `16`，输出 pairwise advantage 和 uncertainty。

```text
静态相对质量：
  native/S7 score logit 与 margin                  3
  native/S7 frozen quality 与 quality margin       3
  pair center/scale/periodic-angle 差异             3
  native/S7 当前 ROI appearance cosine             1
两帧常速度残差：
  native center/scale/periodic-angle/appearance     7
  S7 center/scale/periodic-angle/appearance         7
                                                   --
总计                                               24
```

接管使用保守 lower-confidence-bound：只有 advantage 扣除 uncertainty 后仍越过 promotion margin，且 source 监督表明 S7 明显优于 native 时才允许接管；历史不足、帧不连续、候选缺失、不确定性过高或证据不足时均明确 abstain。状态只保留最近两帧**实际输出的 selected candidate** 的 box 和 ROI embedding；训练和推理使用相同的状态更新语义；周期角度按 OBB 的 `pi` 周期计算最短残差；序列名只用于检测边界并 reset 状态，不进入学习特征。该实现不增加 DINO/RPN/ROI 前向，不保存 dense feature-map history，不引入 optical-flow 网络、跨帧 Transformer、dark expert、EKF 或 Kalman。

为避免视频监督泄漏和因抽取 small 帧而破坏因果上下文，本轮同时锁定：source 训练记录固定按 `split, sequence, frame` 排序；train/validation holdout 按稳定的 sequence 名称检查，同名序列不能跨 split 同时出现；source-small 指标从完整连续 source validation 逐帧结果中切片统计；phase-2 `train_result.json` 仅用于验证 source gate、checkpoint provenance 和固定配置；phase-2 checkpoint 缺少 V2 tiny head 时只允许初始化该新 head，V2 resume 会严格核对 `selective_two_frame`、24 个 scalar channels、hidden=16 和 max-candidates=20。

除原有 source exact-retention gate，V2 还记录并强制检查 selector 自身贡献（只统计 `s7_selective_promotion.promoted=true` 的帧，避免把 phase-2 checkpoint 自身收益误记为 V2 收益）：`same_frame_set`、`selector_top1_loss_zero`、`selector_top1_gain_nonzero`、`selector_small_top1_gain_nonzero`、`selector_gain_multi_sequence`。该 selector-specific 强 gate 只在 V2 flag 开启时生效，不回写历史 V1 的正式裁决。本地检查：`py_compile` 通过、`pytest 151 passed in 1.02s`、`git diff --check` 通过。

服务器结果：训练和 gate 逻辑本身正常，但方法没有学会安全接管——4 个 epoch 均保持 source full `677/738`、small `303/350`、`lost=0`、full/small R@100 `738/738` 与 `350/350`，而 `s7_selective_promotion.promoted=0`，selector 自身 source-small gain 也为零。`best_epoch=0` 是正式 gate 的正确裁决。失败机制：S7 候选能进入候选集合，但 V2 只把 S7 lane top-20 预筛成一个候选，再用 24 个标量判断是否接管；source 中可学习的 gain pair 极少，模型在不确定时始终 abstain，无法利用候选内部的空间细节。裁决：关闭标量两帧 selective promotion V2，但不关闭 `seq03_small` 问题本身。

**（17）stride-7 highres ROI 空间质量读出（阶段 A）**（入口 `crane_project/tools/dino_teacher_s7_highres_roi_ranker_train.py`；`protocol_version=23`）

方法：冻结 DINOv2、native S14 RPN/ROI、S7 readout/RPN 和 affine merge；从已存在的冻结 S7 stride-7 feature map 增加一次轻量 `3x3` output rotated RoIAlign readout，只处理 native top-1 加 S7 lane top-32 候选；用 native semantic ROI embedding、stride-7 spatial ROI embedding、score/geometry/source 标量训练 candidate max-RIoU 与 same-frame relative ranking；推理时只允许经过显式 quality margin 的 S7 候选接管。不增加 DINO forward，不保存 dense feature history，不引入 RGB/前景分支、时序状态、双专家、EKF 或卡尔曼。

只训练 `s7_highres_spatial_projection` 与 `s7_highres_candidate_quality_head`，可训练参数共 `38,465`；`target_dev=null`。

```text
formal native baseline: full 677/738，small 303/350，MCML 3/3
epoch 1:               full 685/738，small 309/350，lost 0
epoch 2:               full 686/738，small 309/350，lost 0
epoch 3:               full 687/738，small 310/350，lost 0，gained 10
epoch 4:               full 687/738，small 310/350，lost 0，gained 10
```

epoch 3/4 的 full/small R@100 分别达到 `738/738`、`350/350`，MCML 保持 `3/3`；full DFR 从 `8.4636%` 改善到 `8.3282%`，ACI 从 `0.82687` 改善到 `0.82837`。S7 成为输出 top-1 的帧数达到 `32`，说明 stride-7 ROI 空间质量读出已经实际利用 S7 候选，不再是 V2 的 `promoted=0`。该结果覆盖此前「最佳 source-safe S7 实验候选」的记录，相对正式 native 基线新增 `10` 个正确帧和 `7` 个 source-small 正确帧，同时保持 `677/677` exact retention，也比历史 affine `688/311、lost=1` 更安全。但它不覆盖正式部署模型，因为预注册 absolute gate 仍有两项各差 1：

```text
full_top1_absolute:  687 < 688
small_top1_absolute: 310 < 311
```

因此 `best_epoch=0` 和 `SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ` 是正式 gate 的正确裁决。epoch 3/4 checkpoint 只能作为冻结诊断候选，不能使用 `labeller_best_source_only.pth` 代替（后者仍是 epoch-0 fallback）。本地检查：新增单元测试与历史 V1/V2 回归测试共 `153 passed`。

**（18）protocol-24 / protocol-25：margin 审计与第一次 fixed target-dev**

不重复训练，只对固定 epoch 3 checkpoint 做一次 source-only、共享模型前向的 promotion-margin 审计（入口 `crane_project/tools/dino_teacher_s7_highres_margin_source_audit.py`）。预注册 margin `0.20、0.225、0.25`；每帧只计算一次 DINO/S7/ROI/quality logits，然后在同一候选输出上离线应用三个 margin；选择规则不变（`lost=0`、full `>=688`、small `>=311`、full/small MCML `<=3`、DFR/ACI 不退化），多个 margin 通过时固定数值最大者。实现额外强制：输入结果必须精确匹配 `protocol_version=23` 的 epoch 3 `687/310、lost=0、gained=10` near-pass；checkpoint 必须是同一 work-dir 下的 `labeller_epoch_03_source_only.pth`；`0.25` margin 必须在本次只读审计中复现 epoch 3，否则立即停止。审计输出 `protocol_version=24`，不更新参数、不复制 checkpoint、不读取 target。

```text
margin 0.20:  full 688/738，small 311/350，lost 0，gained 11，promotion 35
margin 0.225: full 688/738，small 311/350，lost 0，gained 11，promotion 34
margin 0.25:  full 687/738，small 310/350，lost 0，gained 10，复现 epoch 3
```

三组结果来自同一批 `738` 次 frozen model forward 和 `2214` 次 margin decision。按「通过正式 source gate 的最大 margin」规则固定选择 `0.225`；full/small MCML 均为 `3`，full DFR `8.3185%`，ACI `0.82870`，保持 `677/677` exact retention 并新增 `11` 个 full、`8` 个 small 正确帧。因此 epoch 3 + runtime margin `0.225` 是第一个通过正式 source gate 的 S7 实验候选：相对 native 的 `677/303` 保持 `677/677` exact retention 并新增 `11` 个 full、`8` 个 small 正确帧，但 target 尚未读取，正式部署模型不变。另已修复 fallback 结果中 `current_inference_small_validation_summary=null` 的纯报告遗漏，该遗漏不影响既有训练或 gate。

配套一次固定三片段 target-dev（入口 `crane_project/tools/dino_teacher_s7_highres_fixed_target_audit.py`）要求输入 `protocol_version=24` 的 source margin 结果，并同时锁定 epoch 3 checkpoint、checkpoint 架构 margin `0.25`、source 选出的运行时 margin `0.225`，既不伪造 checkpoint 结构，也不允许在 target 上再次搜索 margin。

第一次启动服务器 fixed target audit 时，在读取 target 前被 checkpoint provenance gate 拦截：历史 epoch 3 checkpoint 的嵌套 `s7_highres_roi_ranker` 元数据没有保存冗余的 `frozen_detector=True` 标记，而初版审计脚本误把该标记当作必需字段。该错误只发生在元数据校验阶段，三片段尚未开始推理，不产生 target 结果，也不影响 checkpoint 权重。现已修复为历史 schema 兼容规则：嵌套审计标记缺失时，由 protocol-24 source 审计中的 `dino_parameters_unchanged`、`detector_parameters_unchanged`、只读零参数更新，以及 checkpoint 的 source-only/frozen-DINO、epoch、training mode 和严格 architecture validation 共同证明；若历史 checkpoint 中存在这些嵌套字段，任何与固定设计冲突的值仍会被拒绝。复合错误也已拆分为 source isolation、special routing、extra compute 和 shape 四组。本地检查：相关 `py_compile` 通过，highres 训练、source margin audit、fixed target audit 与历史 V1/V2 及主 labeller 回归测试共 `165 passed`。

`/Users/mac/Downloads/Copy of target_dev_result.json`（`protocol_version=25`）裁决为 `S7_HIGHRES_FIXED_TARGET_DEV_FAIL_KEEP_NATIVE_BASELINE`。source margin gate、checkpoint provenance gate、baseline/native 逐帧复现、参数零更新、DINO 与检测 head 未改变，以及 `candidate_forward=137=40+33+64` 均通过，因此这次失败不是评估索引错位或 target 泄漏。

| 片段 | baseline -> candidate Top-1 | MCML | R@100 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `seq02_far` | `38/40 -> 38/40` | `1` | `40/40` | 未达到预设 `39/40` |
| `seq02_dark` | `29/33 -> 29/33` | `1` | `33/33` | 通过该片段诊断门 |
| `seq03_small` | `50/64 -> 50/64` | `6` | `55/64 -> 64/64` | 候选覆盖通过，Top-1 未改善 |

`seq03_small` 的实际 high-resolution promotion 只有 frame `133、182、186` 三帧，且三帧原本已经 Top-1 正确，因此没有产生新增 Top-1 gain。14 个失败帧中正确候选已经在池内，其候选排名为 rank `2` 有 9 帧、rank `3` 有 2 帧、rank `8`/`12`/`39` 各 1 帧。由此得到的新研究问题不是「stride-7 是否还能产生候选」，而是：在候选已经覆盖的跨域小目标帧上，如何用 source-only 监督学习一个可靠的 native/S7 统一质量排序，使正确候选从 rank 2/3 等位置安全提升到 Top-1，同时保持 native exact-retention，而不是让 high-resolution 分支只在已经正确的帧上 promotion。

**（19）unified native/S7 hard-pair ranker**（`--s7-highres-unified-ranking`；入口 `crane_project/tools/dino_teacher_s7_highres_roi_ranker_train.py`、`crane_project/utils/s7_temporal_association.py`）

方法：DINOv2、native S14 RPN/ROI、S7 readout/RPN、affine merge 和 OBB 几何保持冻结；对同一个 `native top-1 + S7 top-32` active pool 建立统一 fused score，而不是只监督某个 lane winner；用 source GT 的连续 max-RIoU、同帧相对排序和 current-fused-score hard pairs 训练，显式保留 native retention pair 与 S7 gain pair；推理时所有 active candidates 共用一个 rank score，但 S7 仍必须超过 native 的固定 promotion margin 才能接管，零初始化时严格回退 native；source-only 训练期间增加确定性的 feature-domain brightness/blur/scale view 作为跨视图排序稳定性压力。训练、推理和审计复用同一组 quality logits，避免「训练目标与最终接管规则不一致」。

| epoch | full Top-1 | small Top-1 | lost | gained | full/small MCML |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 688/738 | 312/350 | 1 | 12 | 3/3 |
| 2 | 695/738 | 319/350 | 1 | 19 | 3/3 |
| 3 | 696/738 | 320/350 | 1 | 20 | 3/3 |
| 4 | 695/738 | 319/350 | 1 | 19 | 3/3 |

四个 epoch 都只训练 high-resolution projection/quality head，DINO 与原检测 head 参数保持不变，`target_dev=null`。正式 exact-retention gate 的 `best_epoch=0` 不是 checkpoint 保存错误：epoch 1–4 均损失同一帧 `val|real_seq07|215`，因此 `exact_old_correct_retention=false`。epoch 3 是 raw source 最优点，但不能回写成原 exact-retention 协议通过。exact retention 不是所有目标检测研究的通用指标，而是本项目为「替换当前正式模型」预设的风险约束，因此后续采用两级裁决：

1. `formal exact gate` 保持不变，继续决定 `source_safe` 和正式部署候选资格；
2. 新增 `bounded-risk research gate`，只允许在 source full/small、MCML、DFR、ACI 均满足原 gate 其余要求时，容许 `lost<=1`、lost fraction `<=0.002`、gain/loss ratio `>=10`，且净收益至少跨两个 source sequence；它最多授权一次新的固定 target-dev 诊断，不能授权部署、完整 final test 或 source-safe claim。

为避免立刻重训或在 target 上调阈值，先锁定 epoch 3 checkpoint，用同一组 frozen source forward/logits 审计固定 margin `0.25/0.275/0.30`（协议 `protocol-26`，入口 `crane_project/tools/dino_teacher_s7_unified_highres_margin_source_audit.py`，结果 `/Users/mac/Downloads/margin_audit_result.json`）：

| runtime margin | full Top-1 | small Top-1 | lost/gained | full/small MCML | exact gate | bounded-risk gate |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.25 | 696/738 | 320/350 | 1/20 | 3/3 | fail | pass |
| 0.275 | 696/738 | 320/350 | 1/20 | 3/3 | fail | pass |
| 0.30 | 696/738 | 320/350 | 1/20 | 3/3 | fail | pass |

`protocol_version=26` 审计完整复现 epoch 3，`target_dev=null`，DINO 与检测器参数保持不变，参数更新为 0，738 个 source 帧共享一次模型 forward 并产生 `738*3=2214` 个固定 margin decision。相对 native source baseline，full Top-1 从 `677` 增至 `696`，small Top-1 从 `303` 增至 `320`，full R@100 从 `717` 增至 `738`，small R@100 从 `329` 增至 `350`；DFR 从 `0.08464` 降至 `0.07871`，ACI 从 `0.82687` 增至 `0.83910`。收益跨越 real/sim 两个 source sequence，但净收益明显偏向 sim（`real_seq07 +2`、`sim_seq10 +17`）；唯一 source 损失仍为 `val|real_seq07|215`，lost fraction `0.001477`，gain/loss ratio `20`。裁决：

```text
SOURCE_ONLY_UNIFIED_HIGHRES_BOUNDED_RISK_RESEARCH_GATE_PASSED_TARGET_NOT_READ
source_safe=false
eligible_for_fixed_target_dev_diagnostic=true
eligible_for_deployment=false
eligible_for_full_test=false
```

三个 margin 的 promotion 数、Top-1、lost/gained 帧和时序指标均完全相同，说明 `0.25–0.30` 没有跨过新的排序边界，继续扫描全局 margin 不能解决该错误接管；`0.30` 只是 source 指标打平后的保守 tie-break，不代表它优于 `0.25`。另需保留 raw Top-1 `696/320` 与 deployment-threshold Top-1 `694/318` 的差异，正式部署前必须增加逐帧 deployment exact-retention gate。该结果可作为「unified high-resolution 排序在 source 上显著提高候选覆盖与净 Top-1」的正向消融证据，也可报告 exact-retention 与 aggregate gain 之间的安全权衡，但不得写成 source-safe、target 改善、未知序列泛化或部署结果。同批次的 Pairwise Takeover Ranker V2 已完成 source-only 训练但没有通过 source gate，其关键结果与失败原因只保留在下表，不再展开不可行路线。

**（20）protocol-27 / protocol-28 smooth-geometry support audit**（结果 `/Users/mac/Downloads/result.json`）

本次审计没有读取 target，参数更新为 0，738 个 source 帧共享一次 forward，native baseline 逐帧复现通过。关键事实：`full` 上 `sym_kld` 达到 `732/738`、`55` 次 gain、`0` 次 loss；`source-small` 上达到 `348/350`、`45` 次 gain、`0` 次 loss；`gwd` 与 `normalized_gwd` 也在 full/small 上有正净收益。因此几何排序信号并非为零。

原 protocol-27 给出 `SOURCE_ONLY_SMOOTH_GEOMETRY_RANK_SUPPORT_INSUFFICIENT_TARGET_NOT_READ`，原因是它对 `full` 和 `small` 使用同一个 `min_gain_domains=2`、`min_gain_sequences=2`；而当前正式 source-small 划分只有 `sim` 域和 `sim_seq10` 一个序列，这两个条件在该子集上结构性不可满足，属于 coverage gate 设计不匹配，不是模型 forward 或评估索引错误。协议升级为 protocol-28：full 仍保持 `2/2` 覆盖要求，small 使用独立的 `small-min-gain-domains=1`、`small-min-gain-sequences=1`，同时 JSON 显式记录 `coverage_limited=true`。这只允许进入一次 source-only 研究训练，不允许宣称 small 多域/多序列泛化、source-safe、部署或完整 final test。

同时新增几何引导统一排序训练选项（`--smooth-geometry-ranking`）：以 source GT 计算训练期 smooth OBB geometry（默认 `sym_kld`）并形成辅助 pairwise ranking loss；推理时仍只使用 native/S7 候选特征生成的 quality logits，不读取 GT/target，不改变 DINO、native detector、S7 candidate pool 或 native-protected promotion contract。训练前必须先获得 protocol-28 support JSON，support gate 未通过时训练入口拒绝启动。

**（21）protocol-29 native-relative risk residual V1**（`dino_teacher_s7_native_relative_risk_v2`）

方法：在统一 high-resolution quality head 上加入「仅压低错误 S7、保护可用 S7」的 native-relative risk residual；训练和选模均为 source-only，三个固定 target-dev 片段均未读取。

| epoch | source full / 738 | source small / 350 | lost / gained | source gate |
| --- | ---: | ---: | ---: | --- |
| 1 | 696 | 320 | 1 / 20 | fail |
| 2 | 696 | 320 | 4 / 23 | fail |
| 3 | 697 | 322 | 4 / 24 | fail |
| 4 | 699 | 323 | 4 / 26 | fail |

唯一失败项始终是 `exact_old_correct_retention`；`best_epoch=0`，`source_selected_checkpoint` 指向 epoch-0 native fallback（不是 S7 候选），裁决 `SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ`，`target_dev=null`。

机制定位比「训练损失没有下降」更具体：风险 head 的 raw `risk_scale` 在 epoch 1 已变为 `-1.381873e-05`，epoch 2–4 分别约 `-1.378079e-05`、`-1.377574e-05`、`-1.377574e-05`。现有 one-sided 参数化计算 `max(raw_risk_scale, 0)`，负区间导数为零，因而 effective `risk_scale`、risk penalty mean/max/nonzero 全程为零；训练实际等价于未启用 risk residual 的统一 ranker。日志中的 risk-retention active count 仅约 10、12、11、16 个/epoch，而每 epoch 约有 2,781 个 source train frame、约 81k 个错误 S7 候选，监督极度稀疏。

结论与边界：

- 不可把 `699/323` 当作可选 checkpoint：它以 4 个旧正确帧损失换得 gains，违反 exact retention。
- 不可据此宣称 native-relative risk 思路无效：本实现的 gate 已死锁，不能产生任何风险 penalty。
- 也不应仅把 `max(raw,0)` 换成可导函数后直接复训同一方案：正向 active 风险对极稀疏，先前 source gate 的失败机制仍未被结构性解决。
- 下一步限定为只读 source-only native-relative risk / score-gap support audit。

**（22）protocol-30 native-relative risk support audit**

只读 source-only 审计 `Source-only Native-relative Risk Score-gap Support Audit` 已完成。审计使用 protocol-29 的 epoch-0 native fallback，在进程内临时启用已冻结的 S7 candidate readout；共执行 `2,781` 次 source train/train_sim candidate forward，候选数固定为 `33/frame`；`parameter_update_count=0`，DINO 和检测 head 参数均未改变，`target_dev=null`。

| 指标 | 结果 |
| --- | ---: |
| source train frames | 2,781 |
| native-correct frames | 2,774 |
| wrong S7 candidates | 82,291 |
| usable S7 candidates | 6,701 |
| native-wrong / usable-S7 frames | 7 |
| 以上 gain frames 的覆盖 | 仅 sim / `sim_seq08` |
| oracle 为 S7 的 frames | 652 |
| oracle S7 未被 base fused 排为 Top-1 | 601 |
| effective risk penalty nonzero | 0 |

JSON 表面记录 `active_required_penalty_candidate_count=1`、`required_penalty=0.138764`，但该唯一候选来自 `train_sim|sim_seq08|74`：native 已错误（RIoU=`0.370908`），同帧有 2 个 usable S7。protocol-29 的 retention objective 只在 `native_correct and wrong_s7` 时启用，因此该候选不属于 V1 可训练的 retention 正监督；`active_required_penalty_frame_count=0` 正是这一事实的反映。当前汇总字段把 all-frame required penalty 与 native-correct eligible retention support 混在同一层，容易误读；后续应在 schema 中显式分开，但现有 `frame_rows` 已足以重算，不需要再次执行 2,781 帧 forward。

正式裁决：

- epoch-0 上 native-relative risk V1 的 eligible positive retention support 为 `0`；protocol-29 后期每 epoch 约 10–16 个动态 active pair 是 quality head 漂移后才出现的极稀疏信号，不能证明该 residual 有稳定的初始可学习支持。
- 7 个真实 takeover gain frame 全部来自一个 simulation sequence，缺少 real 域与多序列覆盖。
- risk residual 只能压低 S7，不能解决这 7 个「native 错、正确 S7 需要上升」的 gain case；监督方向与剩余问题不一致。
- protocol-29 native-relative risk residual V1 正式关闭；不进行简单 gate 改写后复训，也不进入同目标 V2，不读取固定 target-dev，不运行完整 final test。
- 该结果是有用的负证据：候选覆盖仍然存在，但当前 source 分布没有为「仅抑制错误 S7」提供足够且跨域的正监督。

**（23）protocol-31 paired-view candidate role-switch support audit**（入口 `crane_project/tools/dino_teacher_s7_paired_view_role_switch_support_audit.py`）

方法：锁定 unified high-resolution epoch-3 candidate checkpoint；对每个 source-train/train-sim 原始帧分别运行 `clean`、`photometric`（gamma/exposure/contrast）和 `degradation`（blur/noise/downsample-upsample）三个**图像级**视图；每个视图都重新经过冻结 DINO、native/S7 RPN 和 ROI，不能复用或扰动 cached feature tensor。变换不含 crop、flip、rotation 或其他空间 warp，故原始 OBB 标注保持有效；feature cache key 同时固化 view 名称与版本。source validation 只运行 clean view，并必须复现锁定 native baseline full `677`、small `303` 后才允许解释 train 的支持统计。协议不训练、不选 checkpoint、不读取 target、不产生部署阈值，并将 DINO/检测 head 参数版本在前后比较；同一原始帧在多个增强视图中产生的信号只计为一个 unique support frame。

预注册训练授权门：

```text
paired clean-native -> augmented native-wrong/S7-correct role-switch frames >= 32
gain role-switch 覆盖 real 和 sim
gain role-switch 覆盖 >= 3 个 source sequences
任一 sequence 占 role-switch frames <= 50%
native-retention support 同时覆盖 real 和 sim
```

实际结果为 `SOURCE_ONLY_PAIRED_VIEW_ROLE_SWITCH_SUPPORT_INSUFFICIENT_TARGET_NOT_READ`：

| 指标 | 结果 |
| --- | ---: |
| clean source-val reproduction | full `677/738`，small `303/350`，通过 |
| candidate forwards | `9,081 = 2,781 × 3 + 738` |
| parameter updates | `0` |
| paired gain role-switch 原始帧 | `10 / 32` |
| role-switch 覆盖 | 仅 sim / `sim_seq08` |
| 单一 sequence 占比 | `1.0`，要求 `<=0.5` |
| retention support | `0` |
| target-dev | `null` |

两种 augmented view 并非未生效：photometric/degradation feature cache 均为首次计算，且 degradation 将 train native hit 从 `2,774` 降至 `2,768`、native-wrong/S7-correct 从 `7` 增至 `13`。10 个 role-switch 都是 `sim_seq08` 的临界 RIoU 帧：clean native 刚好正确，而增强后 native 跌破 `0.5`、S7 仍正确。因此它们是单一模拟序列中的人工退化支持，不能作为跨域排序训练集。

该结果同时发现 split 层面的监督错配：protocol-31 train 包含 `real_seq01/04/05/06` 与 `sim_seq08`，其中 real train `2,033` 帧全部 native 正确；clean validation 却有 `57` 个 native-wrong/S7-correct frame，其中 `real_seq07=9`、`sim_seq10=48`。不能把这些 validation frame 直接并入旧训练集（会破坏现有 source gate），也不能放宽 protocol-31 coverage gate 或继续扩大 degradation。

**（24）protocol-32 source sequence cross-fit support feasibility audit**（入口 `crane_project/tools/dino_teacher_s7_sequence_crossfit_support_audit.py`）

JSON-only 入口：只读取 protocol-31 的 `train_frame_rows` 与 `validation_clean_frame_rows`，不读取图像、DINO、checkpoint、target 或标注，也不进行参数更新。按完整 sequence 枚举 leave-one-sequence-out fold，明确禁止随机 frame split；每一 fold 的训练侧必须同时有 real/sim natural gain 支持，held-out 侧必须保留足够的困难 gain frame；正式总 gate 还要求至少一个 viable real hard held-out sequence 和一个 viable sim hard held-out sequence。

结果 `SOURCE_ONLY_SEQUENCE_CROSSFIT_SUPPORT_INSUFFICIENT_TARGET_NOT_READ`，无 DINO forward、无参数更新、`target_dev=null`。7 个 sequence 的 leave-one-sequence-out 中有 `2` 个 viable fold：held-out `sim_seq08` 与 held-out `sim_seq10`；但 `real_seq07` held-out 时训练侧只剩 simulation natural gain，故 `viable_heldout_real_sequences=[]`。总 gate 的 `minimum_valid_folds=true`、sim held-out coverage=true，但 real held-out hard-sequence coverage=false。这说明目前 source 数据无法构成无泄漏的跨域 selector 训练/验证协议；关闭 learned S7 selector，后续应补充至少两个独立标注的真实 source 困难连续序列（其 frozen native 错误、S7 有正确候选），并按完整 sequence 留出验证，而不是使用 target、把 `real_seq07` 直接并回旧训练集，或调低门槛。

**（25）protocol-33 dense object-centric temporal separability**（入口 `crane_project/tools/dino_teacher_s7_dense_temporal_separability_audit.py`）

protocol-32 只否定了依赖稀疏 `native-wrong/S7-correct` 接管标签的现有 selector 训练协议，没有否定利用所有连续 source 标注帧学习对象级时序表示。`seq02_dark` 的有效改进也说明方法可以通过改变监督任务而不依赖分支接管标签：其冻结 DINO 检测头使用的是密集 RPN/ROI source 监督。

方法：锁定 unified high-resolution epoch-3 candidate checkpoint，在 source `train + train_sim` 的完整连续序列上，对 `clean/photometric/degradation` 三个图像级视图分别重新运行冻结 DINO、native/S7 RPN 和 ROI。每帧使用 source GT 将 active pool 候选标为 `max-RIoU>=0.5` 的对象正候选和 `max-RIoU<=0.30` 的高分 hard negative，随后只读统计：相邻帧正确候选相对高分错误候选的 ROI embedding / highres embedding cosine margin；clean 与两个 label-preserving view 之间的对象级 cosine margin；使用前两帧正确候选构造的中心、对数尺度和周期角度常速度残差可分性；按完整 source sequence 分组的 pair 数和成功比例。与已失败的阶段三 temporal student 不同：student 训练同帧连续 RIoU、relative quality 和 teacher distillation 并在固定七标量递归选择器上推理，protocol-33 不训练 selector，而是先判断 frozen ROI/highres 表示是否在多个真实 source 连续序列中具备对象级跨帧、跨视图可分性。

预注册默认支持门：每个 sequence 同时具有至少 `32` 个 temporal pair 和 `32` 个 cross-view pair；二者在固定 cosine margin `0.02` 下的成功比例都不低于 `0.60`；满足条件的 source sequence 至少覆盖 `2` 个 real 和 `1` 个 sim。运动残差完整报告但暂不作为硬门，避免在未验证 camera/crane 运动尺度前用手工权重误杀有效外观信号。

结果 `SOURCE_ONLY_DENSE_TEMPORAL_SEPARABILITY_PASS_TARGET_NOT_READ`：冻结候选前向 `9,081` 次、参数更新 `0`、target 为 `null`，并复现 clean source-val native baseline full `677/738`、small `303/350`。source-train 得到 temporal pair `2,773`、cross-view pair `5,562`；组合外观 margin 成功率为 `0.8990/0.9642`，中位数为 `0.09196/0.11664`，合格完整序列覆盖 real `seq01/04/05/06` 与 sim `seq08`。分解结果表明主要可学信号来自 frozen ROI 1024-D embedding（temporal/cross-view margin 中位数约 `0.1845/0.2322`）；原 highres 32-D embedding 的 cosine margin 近零（约 `0.000049/0.00069`）。`real_seq05` 是 real source 中最困难的完整连续序列，因此后续仅作为内部 sequence holdout，禁止随机拆帧。该结果只授权 source-only adapter 训练，仍未授权读取固定 target-dev。

**（26）protocol-34 ROI temporal contrastive candidate adapter V1**（入口 `crane_project/tools/dino_teacher_s7_roi_temporal_contrastive_train.py`）

方法：锁定 protocol-33 的 unified high-resolution epoch-3 候选基座，冻结 DINO、native/S7 RPN、ROI 与原 highres quality head；只训练 `1024 -> 128 -> 64` 的 L2-normalized ROI projector。监督来自完整 source 连续序列的相邻帧正候选、三个固定视角的跨视角正候选和同帧 top-8 hard negatives，不再依赖稀疏 takeover 标签。训练使用 real `seq01/04/06` 与 sim `seq08`，完整 `real_seq05` 仅用于内部检索指标选 epoch；官方 source-val 只在 epoch 固定后执行一次最终 gate。训练后 holdout 的 margin 成功率和中位数还必须同时不低于 epoch-0 原始 ROI cosine，否则即使后续检测数字偶然通过，也不能把随机投影或退化表示解释为有效学习。实现进一步锁定 temporal/cross-view 两个分支分别非退化，epoch selection 先最大化二者较弱项，再依次比较 temporal、cross-view 成功率及两类中位 margin。

推理：先确定 native fallback，仅当最佳 S7 候选相对 native 的 learned ROI cosine 与常速度 OBB 残差融合优势达到固定 `0.05` 时才接管；断帧或 sequence 边界立即复位。正式 gate 保持旧正确帧 `lost=0`、full `>=688`、small `>=311`、full/small MCML `<=3`，并要求 DFR/ACI 不退化。active pool 上限语义固定为「额外 `1` 个 native + 最多 `32` 个 S7」，native 不占 S7 名额。source 结果还必须分别报告 margin promotion 与所有 S7 state update（包括 native 无有效候选时的 S7 fallback）的连续长度、错误状态连续长度、S7 状态后旧正确帧丢失、恢复间隔和未恢复事件；其中「S7 状态后旧正确帧丢失为 `0`」显式并入 source gate。

8G 显存约束已写入入口：三个视角逐个前向；仅将代表正候选和 top-8 hard negatives 以 CPU FP16 保存到 `roi_temporal_source_train_cache_fp16.pt`；正式 adapter batch 固定为 `128`；legacy SDPA query chunk 最大 `256`；默认每帧调用一次 CUDA cache 释放。该机制限制活跃张量和缓存上界，但 `nvidia-smi` 的 reserved memory 仍可能随算子 workspace 在安全范围内波动。正式 protocol-34 进一步锁定最多三张 8G GPU：两张只承载 frozen DINO shard，一张承载冻结检测 head 与 adapter；该拓扑不是数据并行，adapter 的 data-parallel world size 始终为 `1`，effective batch 固定 `128`，故学习率固定 `3e-4`，不能按可见 GPU 数量线性缩放。正式入口同时锁定 epochs `4`、temperature `0.07`、promotion margin `0.05`、motion weight `0.25`，避免资源调整被误写成新的超参数扫描。

source-only 运行选中 epoch `4`。`real_seq05` 内部 holdout 上，adapter 的 temporal/cross-view 检索成功率达到 `0.989247/0.997321`，中位 margin 达到 `0.709181/0.699611`，明显高于原始 ROI cosine 的 `0.655914/0.839286` 与 `0.07304/0.13698`，故 representation gate 通过。这证明 protocol-33 发现的对象级时序信号确实可以被轻量 projector 学习，而不是随机投影造成的假象。

正式 source-val 闭环选择严重失败：native baseline full/small `677/738`、`303/350`；候选结果仅为 `425/738`、`135/350`，full/small MCML `203/125`，old-correct lost `263`、gained `11`。共发生 `381` 次 S7 promotion，其中 `302` 次错误；最长 S7 状态连续段为 `286` 帧，最长连续错误 promotion 为 `203` 帧，S7 状态之后发生 `261` 个旧正确帧丢失。失败集中于 `sim_seq10`：错误 S7 被写入下一帧 reference 后，错误轨迹在外观相似度与运动一致性上自强化。裁决 `SOURCE_ONLY_ROI_TEMPORAL_ADAPTER_FAIL_KEEP_NATIVE_BASELINE`；target 为 `null`，不授权固定 target-dev、部署或完整 final test。

**（27）protocol-35 ROI temporal counterfactual attribution**（只读反事实审计）

方法：固定 protocol-34 epoch-4 adapter、promotion margin `0.05`、motion weight `0.25`，在同一次冻结候选前向中比较 `closed-loop/native-anchor` 与 `top-32/top-8` 四种策略；参数更新为 `0`，target 为 `null`，不进行 margin scan。`native-anchor` 允许 S7 改变当前输出，但下一帧状态始终由 native fallback 更新，因而不会让错误 S7 递归污染长期 reference。

| 策略 | source full | source small | lost/gained | full/small MCML |
| --- | ---: | ---: | ---: | ---: |
| native baseline | `677/738` | `303/350` | `0/0` | `3/3` |
| closed-loop top-32 | `425/738` | `135/350` | `263/11` | `203/125` |
| native-anchor top-32 | `658/738` | `284/350` | `35/16` | `8/8` |
| closed-loop top-8 | `415/738` | `144/350` | `274/12` | `130/130` |
| native-anchor top-8 | `663/738` | `290/350` | `31/17` | `5/5` |

native-anchor 将 S7 state update 与 post-S7-state old-correct loss 都降为 `0`，同时把 full 从 `425` 恢复到 `658/663`，因此确认 protocol-34 的主要灾难性退化来自递归状态污染。然而最佳 native-anchor top-8 仍比 native baseline 少 `14` 个 full Top-1、少 `13` 个 small Top-1，并丢失 `31` 个旧正确帧；top-8 也没有稳定优于 top-32 的闭环证据，故候选池范围失配不是根本原因。机制裁决：

> protocol-33 正确证明了 source ROI 表征具有对象级时序可分性；protocol-35 进一步证明，这种可分性不足以直接构成 native–S7 takeover quality criterion。主要原因是对象身份/轨迹一致性与候选相对几何质量并不等价：连续错误背景同样可以形成稳定时序表征。递归状态隔离能够消除长链漂移，却不能解决单帧错误接管。

因此关闭基于 ROI temporal cosine 的同构 takeover selector，不继续重训、扫描 margin 或读取 target。该结论不否定 protocol-33 的表征证据，而是限定其适用范围：它可支持对象级关联表示，不能在缺少可验证 real source takeover supervision 时被直接解释为安全的 native–S7 质量排序器。

**门槛与保留状态汇总**

「回退到 `best_epoch=0`」只表示**正式 exact-retention gate** 的裁决，不等于该实验在 source 上没有任何通过的指标，也不等于整个方向被否定。三类状态必须分开读：

| 实验（协议） | source exact gate | bounded-risk gate | 三段 fixed target-dev | 最终保留状态 |
| --- | --- | --- | --- | --- |
| Pairwise V1 / V2 | fail（retention 未达标） | 未定义 | 未运行 | 不采用 |
| S7 第一版 readout/RPN | fail（如 654/677） | 未定义 | 未运行 | epoch 0，S7 disabled |
| retention-aware affine merge | fail（676/677，lost=1） | 未定义 | 运行过一次（宽松 gate 通过，仅诊断） | 不作部署；epoch 1 仅留作后续实验起点 |
| lane arbitration v1 / v2 | fail（lost 4–30） | 未定义 | 未运行 | epoch 0 |
| non-positive quality suppression v1 | fail（risk pair 0/2781，no-op） | 未定义 | 未运行 | epoch 0 |
| 统一时序关联 v1（六 cue） | fail（680/738、305/350，`lost=0`） | 未定义 | 未运行 | epoch 0 |
| 候选级连续 RIoU quality head | fail（681/738、306/350，`lost=0`） | 未定义 | 未运行 | epoch 0 |
| 阶段二B relative-quality | **pass**（691/738、312/350、`lost=0`、MCML 3） | 未定义 | 已授权并执行一次 | `best_epoch=4`；作为阶段三 teacher |
| 阶段三候选排序 student | pass（`best_epoch=4`） | 未定义 | 运行一次 → fail（`seq03_small` 50/64） | source-safe 但不作部署，不授权完整 test |
| 静态域泛化 ranker（protocol-22） | fail（raw 最优 695/738、319/350，lost=1） | 未定义 | 未运行 | epoch 0，关闭 |
| selective promotion V1 | fail（full/small 绝对门 + 多序列净增益） | 未定义 | 未授权 | epoch 0 |
| selective promotion V2（标量两帧） | fail（`promoted=0`） | 未定义 | 未授权 | epoch 0 |
| stride-7 highres ranker（protocol-23/24） | **pass**（margin `0.225`：688/738、311/350、`lost=0`） | — | 运行一次 → fail（`seq03_small` 50/64） | 冻结诊断候选，正式模型不变 |
| unified hard-pair ranker（protocol-26） | fail（lost=1） | **pass**（696/738、320/350） | 授权但未执行（无结果记录） | 研究性消融证据，`source_safe=false` |
| native-relative risk residual V1（protocol-29） | fail（lost≥1） | 未定义 | 未运行 | epoch 0，关闭 |
| ROI temporal contrastive adapter V1（protocol-34） | fail（425/738、135/350、lost=263） | 未定义 | 未授权 | 保留 native baseline |
| 联合前端 `crane_symeood_dino_unified_v1.py` | 未运行（`joint_source_gate_required=True`） | — | 未运行 | 仅实现、未验证 |

全过程中**没有任何分支通过三段 fixed target-dev**，因此正式 deployable 状态始终是 native S14 `alpha=0.5`、S7 disabled。

<a id="sec-3-2-5"></a>

#### 3.2.5 三段 fixed target-dev 的当前证据

这三段只是固定诊断切片，不能拼成一个模型的总成绩，也不是未知序列或部署测试。下表只保留能界定当前问题的结果，不同方法的历史数字明确隔离。

| 切片 | 当前 highres ROI（protocol-25） | 不可合并的历史参照 | 保留结论 |
| --- | --- | --- | --- |
| `seq02_far`，40 帧 | `38/40 -> 38/40`，R@100=`40/40` | temporal student 曾为 `38/40 -> 39/40`，是另一 source-safe 模型 | 当前 highres 未提高 Top-1；far 不再是小目标主线的候选生成瓶颈 |
| `seq02_dark`，33 帧 | `29/33 -> 29/33`，R@100=`33/33` | 无可替代的 Top-1 正增益 | 暗光片段作为保护性诊断；不能称为已完全解决 |
| `seq03_small`，64 帧 | `50/64 -> 50/64`，R@100=`55/64 -> 64/64` | global affine 曾为 `51/64`，但 source `lost=1`，不可部署 | 候选覆盖已补足，正确候选仍未被安全排到 Top-1，是当前唯一明确的研究瓶颈 |

当前唯一可写的跨域诊断结论是：**S7 已解决 `seq03_small` 的候选覆盖，但没有解决最终质量排序。** 历史 global affine 的 `51/64` 只能作为非部署诊断上界，temporal student 的 `seq02_far=39/40` 与当前 highres 不是同一模型，这些数字不能合并成一个已完成的结果。

**MCML 口径必须统一**：当前固定 `seq03_small` RIoU top-1 诊断记录为 `MCML=6`，另有完整组合/系统 evaluator 口径报告 `MCML=14`；后续结果必须同时明确 RIoU 阈值、score threshold、silence 处理、候选过滤和序列 reset 规则，不能混用。

<a id="sec-3-2-6"></a>

#### 3.2.6 支持审计族与根因结论

`seq03_small` 最初确有空间采样瓶颈，短边约 `1.124` 个 DINO token、S14 RPN@2000 为 `45/64`；但 anchor 覆盖 `63/64`、assignment `64/64` 排除了「理论 anchor 不存在」。朴素多尺度可把 RPN 上界提高到 `56/64`，但会伤害 far/dark，故不可部署。高分辨率 S7 已证明可补足最终候选覆盖：protocol-25 的 source-safe high-resolution ROI readout 在 source 达到 full/small `688/738`、`311/350`、`lost=0`；固定 target-dev 中 `seq03_small` R@100 `55/64→64/64`。同一 protocol-25 下 far `38/40`、dark `29/33`、small `50/64` 均无 Top-1 增益；small 的 14 个失败帧中正确候选已位于 rank `2`（9 帧）、`3`（2 帧）、`8/12/39`（各 1 帧）。S7 promotion 仅发生在原本已经正确的 3 帧。因此当前 target 失败是**最终质量排序**，而不是候选缺失。

最根本的问题是 source 监督可辨识性与 split 支持错配：现有 source-train 的 real `2,033` 帧全部 native 正确；natural native-wrong/S7-correct gain 只在 `sim_seq08` 有 `7` 帧。相反，旧 validation 中有 `57` 个自然 gain（`real_seq07=9`、`sim_seq10=48`），不能直接并入旧训练而不破坏 source gate。protocol-31 的完整图像级 paired-view 只产生 `10` 个 role-switch，仍全部来自 `sim_seq08`；protocol-32 的 sequence cross-fit 没有任何 viable real held-out fold。

对象级时序可分性不等价于安全的 native–S7 接管质量：protocol-33 正确证明了 source ROI 表征在 GT 正候选锚定下具有跨帧、跨视图对象级可分性；但 protocol-34 的闭环选择发生递归错误状态污染，protocol-35 即使改为 native-only 状态锚定，最佳 source 也仅为 full/small `663/738`、`290/350`、`lost=31`，仍低于 native baseline。因此时序身份一致性不能直接充当候选质量或 takeover criterion。

**早期时序重锚失败的适用范围**：早期 BrightAug 手工时序重锚失败时，暗光正确候选位于约 top-5708，正确候选排在极深位置；当前 S7 已将小目标正确候选推进到 R@20 `63/64`、R@100 `64/64`，候选池结构已经不同，因此不能把早期失败直接外推到当前候选池。但这也要求后续实验采用 source-only 预检和固定停止门，避免重新扫描 target 权重。该路线本身（去 teacher-force 的手工时序重锚）已封存。

**根本问题的准确表述**

> 在 `seq03_small` 所代表的未知小目标场景中，正确 S7 候选已经能够进入最终候选池，但现有 source 数据**没有同时提供**跨 real/sim、跨 sequence 的「native 失败而 S7 正确」训练证据，以及独立 real hard sequence 的留出验证证据。因此当前无法从既有 source split 中识别、训练并证明一个既能提升正确候选、又保持 native exact-retention 的统一质量排序/接管器。

这一定义区分了三件事：`seq03_small` 是固定 target-dev 上的症状与诊断载体，不是训练样本；S7 候选覆盖是已取得的结构性进展；未解决的是安全跨域排序所需的 source supervision，而不是继续扫描 margin、扩大 S7 RPN 或叠加同类后处理。

**实验族归纳**

| 证据组 | 归纳结论 | 对当前问题的作用 |
| --- | --- | --- |
| 采样与候选审计 | S14 的小目标候选覆盖不足，但 anchor 不缺；高分辨率 S7 可将 fixed small R@100 补到 `64/64` | 排除「继续加 anchor / 盲目扩大 RPN」作为根本解 |
| fixed target-dev 归因 | correct candidate 已在 14 个失败帧的 top-100 内，但未升至 Top-1 | 将问题定位到最终质量排序 |
| exact-retention 实验族 | affine、lane、static/unified ranker 的 source 原始收益均伴随 old-correct loss；少数保持 exact retention 的版本（阶段二B、protocol-23/24）虽有 source small Top-1 增益，但在 target-dev 上仍未提高 Top-1 | 安全约束不是可放宽的超参数，而是部署前提 |
| 时序与 student | 可维持 source safety、候选覆盖或 far 增益，但 small Top-1 仍为 `50/64` | 不将其他模型的 far gain 误记为 small 解决 |
| support audits（risk、paired-view、cross-fit） | 既有 source-train 没有分散的 real/sim gain 支持，也没有 viable real hard holdout | 当前根因是监督/验证支持不足，不是 head 复杂度不足 |
| 系统审计 | DINO teacher 延迟 `1.74–1.98 s/frame`，最终泛化与统一部署均未完成 | 结构有效后才进入统一、蒸馏和真实未知序列验证 |

**已经做过、不再重复的动作**

| 证据类别 | 已保留的结论 | 当前裁决 |
| --- | --- | --- |
| 安全部署与固定推理 | FC interpolation `alpha=0.5`、rotated NMS IoU=`0.5` 已由 source exact-retention 选择 | 保留为唯一正式部署基线 |
| 空间采样/候选覆盖 | anchor 并非主因；S7/high-resolution 可提高小目标候选覆盖，但朴素 `[7,14,28]` 多尺度会伤害 far/dark | 保留「高分辨率候选可行」，关闭朴素多尺度 |
| 高分辨率 target 诊断 | protocol-25 将 small R@100 `55→64/64`，但 Top-1 仍 `50/64` | 覆盖问题完成；不能声称检测或泛化完成 |
| 直接排序 / merge / promotion | ROI small-hard、pairwise v1/v2、affine、lane boost/replay、static ranker、selective promotion V1/V2、smooth-geometry、native-relative risk 均不能同时获得收益与 exact retention | 不再重复调 loss、margin、boost、score residual 或同构 ranker |
| 时序 / 学生统一路线 | source-safe 时序与 candidate student 可维持覆盖，但 fixed small Top-1 未提高；ROI temporal contrastive adapter 甚至破坏 source gate | 不把 temporal far gain 与其他模型拼接；该路线不再解决当前核心问题 |
| 风险抑制与支持审计 | non-positive risk、relative-risk、smooth geometry、paired-view、cross-fit 均未给出跨域可训练正支持 | 关闭在既有 split 上继续训练 selector |
| 工程后续 | 统一入口、self-contained checkpoint、蒸馏/加速尚未完成 | 等获得有效且可验证的统一结构后再推进 |

<a id="sec-3-2-7"></a>

#### 3.2.7 未闭环问题与行动顺序（DINO/S7 支线）

**本节全部结论只适用于「冻结 DINOv2 + SymEOOD 检测融合」支线**，不适用于 OBB 观测主线：Base V3 的图像→OBB 真实在线推理入口与固定 TEST 指标链已在 OBB 主线实现并冻结为报告，见 [OBB 观测可靠性与连续输出](OBB观测可靠性与连续输出.md) 第 4 节。某个配置尚未验证，不代表整个项目没有完成统一推理或 TEST。

**四个未闭环问题（支线内）**

1. **训练/推理流程未统一（指本支线）。** 本支线的 S7/时序/接管实验依赖多个训练、审计入口和结果 JSON 传递 checkpoint provenance；尚无一条覆盖「原始图像 → DINO/S7 候选 → 最终 OBB」的统一入口，也没有一份自包含 checkpoint。注：`crane_symeood_dino_unified_v1.py` 已提供联合推理结构，但它本身尚未通过联合 source gate（见 6.2.1）。
2. **DINO 与最终检测模型未形成训练闭环。** 冻结 DINOv2 是当前实验的合法方法边界，但目前只有大 teacher + 分阶段轻量 head，没有统一可训练 student、特征蒸馏或最终导出流程。
3. **实时性不达标（本支线）。** 三张 GTX 1080 上 DINO 分支仍需 `1.74–1.98 s/frame`；DINO 前向是主瓶颈，仅优化 NMS、S7 ranker 或后处理不能达到实时要求。该数字是 DINO teacher 分支的延迟，不是 OBB 观测层的运行开销。
4. **最终泛化证据不足。** 固定三切片已多次参与诊断，不能充当未知域 final test；历史结果 JSON 和 manifest 也必须从最终推理依赖中移除。本支线尚未在被授权前执行完整 final test。

此外仍有一个必须先解决的核心检测问题：`seq03_small` 所代表的未知小目标正确候选虽已进入 top-100，却尚不能稳定排到第一；而 protocol-30–32 已说明现有 source split 不足以训练并留出验证一个跨 real/sim 的安全接管器。因此下一阶段的前提是补齐真实 source hard-sequence 支持，而不是继续修改当前 selector。

**当前固定行动顺序（本支线）**

```text
A. 若仍要解决 seq03_small，只允许结构不同的通用小目标 native-protected V2
   （不得重复 V1 / 标量两帧 V2 / highres unified / risk residual / paired-view / cross-fit / adapter）
B. 统一训练/推理入口 + 自包含最终 checkpoint（本支线）
C. DINOv2-L/14 teacher -> 轻量 student 蒸馏与速度优化
D. 真正未知序列泛化验证
E. 冻结模型的一次性完整 final test，统一导出常规指标及
   R_center / DFR / TDR_w10 / ACI / MCML
F. 自适应协方差 EKF/卡尔曼观测器（大论文后半部分，不属于本支线）
```

若 A 的 source `promotion` 仍为 `0`、`lost>0` 或其他 source 硬安全门失败，不读取 target，不再通过增加 epoch 重复同一实验。若 source gate 通过但一次 fixed target-dev 仍为 `seq03_small=50/64`，关闭当前候选接管路线，不围绕这 64 帧继续调阈值，论文中如实报告候选覆盖与排序之间的上限差距。

**支线裁决（截至 2026-08-08 记录）**

三个层面必须分开写，逐项依据见 [3.2.4](#sec-3-2-4) 门槛汇总表。

**（1）source gate：部分实验通过，不是全部失败**

| 结果 | 实验 | 关键数据 |
| --- | --- | --- |
| 通过正式 exact-retention source gate | 阶段二B relative-quality | `best_epoch=4`：full `691/738`、small `312/350`、`lost=0`、MCML `3/3` |
| 通过正式 exact-retention source gate | protocol-23/24 stride-7 highres ranker（epoch 3 + runtime margin `0.225`） | full `688/738`、small `311/350`、`lost=0`、MCML `3/3`，full DFR `8.3185%`、ACI `0.82870` |
| 通过 bounded-risk research gate（**不授权** source-safe 声明） | protocol-26 unified hard-pair ranker | full `696/738`、small `320/350`、`lost=1`、lost fraction `0.001477`、gain/loss ratio `20` |
| 未通过任何 source 门 | Pairwise V1/V2、S7 第一版、affine merge、lane arbitration v1/v2、quality suppression v1、统一时序关联 v1、quality head、static ranker、selective promotion V1/V2、native-relative risk residual V1、ROI temporal contrastive adapter | 各项数据见 3.2.4 对应记录 |
| 未运行 | 联合前端 `crane_symeood_dino_unified_v1.py` | `joint_source_gate_required=True`，无结果 |

**（2）三段 fixed target-dev：没有任何分支通过**

- **运行过并失败**：protocol-25 的 stride-7 highres 诊断（`seq02_far` 仍 `38/40`，未达预设 `39/40`；`seq03_small` 仍 `50/64`）；阶段三 student（`seq02_far` `39/40` 与 `seq02_dark` `29/33` 通过，但 `seq03_small` `50/64` 未达预注册 `51/64`）。
- **通过宽松诊断门但不可部署**：affine merge epoch 1 的一次配对诊断（`decision = S7_RELAXED_TARGET_DEV_DIAGNOSTIC_PASS`，`seq03_small` `50/64 → 51/64`），但其 `eligible_for_deployment=false`、`eligible_for_final_test=false`，且 source `lost=1`；该门的定义与 protocol-25 的预注册 formal gate 不同，不能互相替代。
- **授权但未执行**：protocol-26 的 bounded-risk 诊断，无结果记录。

因此 `seq03_small` 的最后有效证据仍为 Top-1 `50/64`、MCML `6`、R@20 `63/64`、R@100 `64/64`。

**（3）最终采用状态：没有任何分支替换正式基线**

```text
正式 deployable 模型 = native S14 alpha=0.5、S7 disabled（best_epoch=0 回退），
用于本支线的正式推理与论文基线；
S7 已把固定 seq03_small 的 R@100 候选覆盖提高到 64/64，但最终 Top-1 仍为 50/64；
通过 source gate 的两个 checkpoint（阶段二B epoch 4、protocol-23/24 epoch 3 + margin 0.225）
以及通过 bounded-risk gate 的 protocol-26 epoch 3，只保留为冻结诊断候选；
不得用 labeller_best_source_only.pth 代替这些候选，也不得把它们的 source 指标写成 target 或部署结果；
下一步只允许结构不同的 source-only 小目标实验，之后才做工程闭环、轻量化、
未知序列泛化和一次性完整 final test；
EKF/卡尔曼留到大论文后半部分。
```

---

## 4. 文献依据与适用边界

<a id="sec-4"></a>

本节文献是设计与协议的**依据**，不是本项目已取得的实验结果。原始检索只收录已由 CVF、AAAI、IJCAI、OAAI 或 arXiv 原始页面核实的英文论文。每篇论文只支持其实际覆盖的设计选择，不把点跟踪、世界模型、光流、卫星视频检测或多视角 3D 检测直接等同于本项目的 OBB 候选排序。

<a id="sec-4-1"></a>

### 4.1 方法协议与结构依据

| 论文 | 可借鉴证据 | 本项目用途 / 已实现程度 | 限制 |
| --- | --- | --- | --- |
| **DINO Teacher**，CVPR 2025，*Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection* | 冻结 DINOv2 backbone，只用 source 标注训练下游 labeller；target 不参与 labeller checkpoint 选择 | 已实现「冻结 DINOv2 + source-only oriented RPN/ROI labeller」；**未复现** target pseudo-label student 与完整 feature-alignment student，论文中不能称为 DINO Teacher 全流程复现 | 原论文是完整域自适应流程 |
| **LiDeRe**，CVPR 2026, pp. 2959–2971，*A Lightweight Readout for Fast and Data-Efficient Dense Prediction* | 在冻结大型视觉 backbone 上训练轻量 interpolation+attention dense readout，以较低参数/数据成本恢复细粒度空间预测，覆盖 object detection | S7 结构方向依据。当前只借鉴「冻结 backbone + 轻量 dense readout」原则，未实现完整 readout/attention 结构；S7 第一版已取得候选覆盖正证据 | 完整升级推迟到 source-safe 合并成立之后 |
| **DEIMv2 / Real-Time Object Detection Meets DINOv3**，arXiv 2025, `arXiv:2509.20787` | Spatial Tuning Adapter 将单尺度 DINO 特征转为多尺度细节特征，并强调实时检测的性能/成本折中 | 支持「冻结/预训练语义 + 轻量空间适配」的结构合理性；作为未来压缩或正式 readout 升级后的结构参考 | 使用 DINOv3 与 DETR/DEIM 系列，不是当前 DINOv2 + rotated RPN/ROI 的直接代码模板 |
| **FeatUp**，ICLR 2024 | 学习式上采样低分辨率 backbone 特征，提高小目标的高分辨率特征定位能力 | 高分辨率特征备选依据 | 训练和显存成本可能高于当前所需的 2x stride-7 readout；只有轻量 readout 仍不足时才考虑 |
| **ESOD**，2024，*Efficient Small Object Detection on High-Resolution Images* | 复用原 backbone 做 feature-level object seeking，只对稀疏目标区域切高分辨率 patch | 若全图 S7 readout 无法满足 GTX 1080 实时预算，可作为稀疏高分辨率候选分支 | 仍应复用同一 DINO，而不是新增独立小目标模型 |
| **DCFL / AI-TOD-R**，2024，*Oriented Tiny Object Detection* | 旋转微小目标的动态 prior 和 coarse-to-fine assignment | 只有新 S7 特征形成后仍出现 assignment/objectness 失配时才引入其动态先验思想 | 当前理论 anchor 覆盖已经很高，不应先于 dense readout |
| **RoMa**，CVPR 2024 | 冻结 DINOv2 提供稳健但粗糙的语义特征，专门的 ConvNet fine features 补充精确定位 | 旁证 | dense matching 而非旋转目标检测；新增 fine encoder 增加复杂度 |
| **Guided Distillation**，WACV 2024 | 冻结/预训练 backbone 下的下游检测分割工程经验 | 旁证：其核心贡献是半监督 guided burn-in | 不直接解决 stride-14 旋转小目标 RPN 问题 |
| **Object-DINO**，2026，*Finding Distributed Object-Centric Properties in Self-Supervised Transformers* | 对象信息分布在 Transformer 多层和 q/k/v patch interaction 中，而不只在最后一层 token | 暂缓；不与 LiDeRe readout 同时实现 | 需要访问全层注意力并聚类 heads，推理成本和当前 rotated RPN/ROI 接口跨度都很大 |

<a id="sec-4-2"></a>

### 4.2 候选排序、时序关联与读出

| 论文 | 可借鉴证据 | 本项目用途 | 限制 / 风险 |
| --- | --- | --- | --- |
| **TQ-01** Pixel-level Quality Assessment for Oriented Object Detection，AAAI 2026, 40(16): 14005–14013, DOI `10.1609/aaai.v40i16.38411` | 旋转检测最终依赖候选排序，分类分数不必然反映定位质量；用像素级空间一致性替代结构耦合的 box-level IoU prediction | 把「candidate quality 必须与真实 OBB 定位质量对齐」作为方法动机和相关工作 | BrightAug PQA v1–v3 已正式失败，该论文**不能授权 PQA v4**；若在新候选空间使用质量信息，必须先做独立预检，并优先采用排序/关联证据，而不是 quality-only top-1 |
| **TQ-02 RARE**，CVPR 2026, pp. 11556–11566 | 把 confidence estimation 作为相对质量排序问题，联合 point-wise quality alignment 与 pair-wise ordering，再从多候选中 retrieve 最优解 | 当前 `64/64` 候选池更需要 relative rank-and-retrieve，而不是全局 affine 或 frame-wide delta | 原任务是单目 3D DETR；价值在损失和候选检索思想，不是网络结构 |
| **TT-01 Dist-Tracker**，CVPR Workshops 2025, pp. 6667–6675 | 小目标轻微位置抖动会使 IoU 关联不稳定；FLIT tracker 联合 L2 与 IoU，用中心距离补偿位置不确定性 | `seq03_small` 的关联不能只看 rotated IoU，应至少联合归一化中心距离；`seq02_dark` 大目标仍可让 IoU 保持较大权重 | 场景是红外多 UAV，且为 workshop/challenge 工作 |
| **TT-02 Chrono**，CVPR 2025, pp. 1962–1972 | 在 DINOv2 上加入 temporal adapter，使冻结表征获得长期时序感知，能以简单特征匹配完成高质量点跟踪 | 复用 DINO ROI/patch embedding 做跨帧外观一致性，而不是另训一个依赖 target 片段身份的路由器 | point tracking 非 OBB；优先借鉴 temporal feature adapter 与 matching 表征，不照搬输出头 |
| **TT-03 COVTrack**，ICCV 2025, pp. 10054–10063 | 连续标注轨迹对稳健 tracker 学习重要；按帧内和帧间置信动态融合 appearance、motion 和 semantic cues | 支持统一使用分类/质量、中心运动、旋转几何和 DINO 外观多线索，而不是固定一套 IoU 权重或按暗光/小目标切换 | OVMOT 结构过重；当前单类/近单目标场景应提炼为轻量因果 scorer |
| **TT-04 VOVTrack**，ICCV 2025, pp. 7472–7482 | 利用无标注原始视频构造 self-supervised object similarity learning，服务跨帧关联 | 若存在严格隔离的无标注 target-train 视频，可学习跨帧 DINO ROI similarity，不需要 target GT | 必须先证明 target-train 与 dev/test 隔离；否则只允许在 source 连续序列训练 |
| **TP-01** Leveraging Temporal Cues for Semi-Supervised Multi-View 3D Object Detection，CVPR 2025, pp. 27401–27412 | 单纯置信阈值不足以保证伪标签定位质量；使用前向/反向预测集成、tracking infill 和辅助检测头过滤改善时序伪标签 | 正式 target-train 阶段不应按单帧 S7 分数生成伪标签，应要求跨帧持续、native/S7 一致或辅助候选质量证据 | 原任务为多视角 3D；只借鉴伪标签过滤原则 |
| **TP-02 PWOOD**，CVPR 2026, pp. 27644–27654 | Orientation-and-Scale-aware Student 与 class-agnostic pseudo-label filtering 降低旋转检测对固定伪标签阈值的敏感性 | 正式学生训练应避免单一固定 score threshold，并显式保护 orientation/scale | 使用部分弱标注，不等同于本项目纯无标签 target-train |
| **TT-05 Tracktention**，CVPR 2025, pp. 22809–22819 | 显式 point tracks 可作为轻量插件改善视频模型的时序对齐和长程一致性 | 若短期 box association 仍不稳，可用内部 DINO point tracks 补充 motion cue | 额外 tracker 的误差会传播；不应作为第一版依赖 |
| **TT-06 MOTIP**，CVPR 2025, pp. 27883–27893 | 以历史 trajectory object features 为上下文直接预测当前检测 ID，可替代复杂手工 cost matrix | 若轻量多线索 scorer 的 source 上界不足，再考虑小型 trajectory decoder | 当前单目标任务不需要完整 ID dictionary/多目标系统 |

**证据—论点映射**

| Source ID | Abstract-level finding | Supported claim | 用途位置 | 风险 |
| --- | --- | --- | --- | --- |
| TQ-01 | OBB 分类分数与定位质量可能错位，像素级一致性可估计质量 | 当前正确 S7 候选不能只按分类分数选 top-1 | Related Work：OBB quality ranking | PQA v1–v3 已失败，只用于动机/新池预检 |
| TQ-02 | 相对 ranking+retrieval 可改善候选质量顺序 | 应学习候选间相对顺序而非 frame-wide delta | Method：candidate evidence | 原任务是 3D DETR |
| TT-01 | 小目标 IoU 因位置抖动失稳，L2+IoU 更稳 | 小目标时序关联必须加入中心距离 | Method：association cost | workshop/红外 UAV |
| TT-02 | DINOv2 + temporal adapter 可形成时序感知特征 | DINO ROI embedding 可承担跨帧外观 cue | Method：appearance cue | point tracking 非 OBB |
| TT-03 | 连续视频需自适应融合 motion/semantic/appearance | 暗光与小目标可用统一 multi-cue selector | Method：unified causal selector | OVMOT 结构过重 |
| TT-04 | 无标注原始视频可自监督学习 object similarity | 隔离 target-train 可用于无 GT 时序关联学习 | Training：target-train consistency | 数据隔离是硬门 |
| TR-01 / LiDeRe | 冻结 backbone 的轻量 readout 可恢复 dense detail | S7 候选生成方向有文献依据 | Related Work：frozen dense readout | 当前 coverage 已足够 |
| TP-01 | tracking infill / 多证据过滤改善时序伪标签 | 正式伪标签不能仅用单帧 S7 confidence | Training：pseudo-label filtering | 3D 到 2D 迁移 |
| TP-02 | OBB student 与 class-agnostic filtering 降低固定阈值敏感 | 学生训练应保护方向/尺度并减少静态阈值依赖 | Training：oriented student | 弱标注设定不同 |

<a id="sec-4-3"></a>

### 4.3 时序表示与排序损失

| 论文 | 可借鉴证据 | 本项目用途 | 限制 |
| --- | --- | --- | --- |
| **Back to the Features: DINO as a Foundation for Video World Models**，arXiv 2025, `arXiv:2507.19468` | 在冻结 DINO 特征空间中学习未来特征/动态，而不是重新训练大型视觉编码器 | 支持在 frozen DINO ROI embedding 上学习短时运动残差，避免重复 DINO 前向 | 视频世界模型，不是 native–S7 候选接管器 |
| **Temporally Consistent Object-Centric Learning by Contrasting Slots**，CVPR 2025 | 使用冻结 DINOv2 逐帧特征，通过对象级对比目标学习时序一致性 | 支持在 source 视频上增加对象级一致性/对比监督，而不是使用序列名称或固定帧段规则 | 无监督对象中心 slot 与有监督 OBB 排序不同，不能原样移植网络和损失 |
| **Optical Flow Estimation for Tiny Objects**，IJCAI 2025 | 专门讨论 tiny object 的运动估计，报告小于 `1%` 的额外参数开销 | 支持「极小目标需要专门运动残差」，并为轻量化设计提供参照 | 当前不引入完整光流网络；先用 box/embedding 的常速度残差验证最低成本版本 |
| **Learn Temporal Consistency for Robust Satellite Video Detector**，arXiv 2026, `arXiv:2606.15112` | 时序特征聚合、结构编码和一致性约束用于遥感视频细粒度旋转目标 | 与本项目旋转、小尺寸、跨帧尺度/角度/结构变化最接近；借鉴中心、尺度和周期角度 residual | 完整特征聚合模块更重，数据域不同 |
| **LSTT: Long-Short Frame Temporal Transformer for Video Object Detection**，ESWA 302 (2026) | 通过长短时特征和自适应 query 利用视频上下文 | 作为后续长短时信息消融的上界参考 | 完整 Transformer 增加显存、延迟和实现复杂度 |
| **Learning to Rank Proposals for Object Detection**，ICCV 2019 | proposal ranking | unified hard-pair 排序设计依据 | 设计依据，不是本项目实验结果 |
| **RankDetNet**，CVPR 2021 | ranking constraints for object detection | 同上 | 同上 |
| **Rank & Sort Loss for Object Detection and Instance Segmentation**，ICCV 2021 | 直接优化排序 | 同上 | 同上 |
| **Danish et al.**，CVPR 2024，*Improving Single Domain-Generalized Object Detection: A Focus on Diversification and Alignment* | 单源域泛化检测的 view diversification / alignment | 支撑「通用 brightness/blur/scale 视图压力」而非 target 依赖 | 同上 |

<a id="sec-4-4"></a>

### 4.4 综合裁决

这些论文共同支持一个收缩后的结论：**继续复用冻结 DINOv2，以极小的候选级模块补足静态或时序排序，而不是增加条件化双专家、第二套 backbone、光流主干或跨帧 Transformer。**

```text
P0：固定 native+S7 候选，不训练检测器
P1：在 source 连续序列训练/拟合轻量 multi-cue 关联或统一排序
    cues = calibrated score + normalized center distance + rotated IoU
           + log-scale/periodic-angle consistency + DINO ROI similarity
P2：native-safe override、reset 和 exact fallback
P3：source exact-retention + source temporal MCML/DFR/ACI gate
P4：全部冻结后只做一次固定 target-dev diagnosis
```

第一版应以 Dist-Tracker 的 `L2+IoU` 和 COVTrack 的 multi-cue fusion 为核心，Chrono 只提供 DINO temporal feature 依据。PQA/RARE 只在现有 score + association 仍无法从 top-100 候选中选出稳定轨迹，且 source 候选排序预检显示明确监督支持时，才授权独立的 candidate-level relative ranking；不得使用 quality-only top-1。

文献不支持再次训练 lane-wide quality suppression，也不支持把旧 PQA 换名重跑。必须继续生效的负结果：BrightAug 的 RegQuality/PQA v1–v3、QFL、去 teacher-force 手工时序重锚均已封存；新文献只能用于当前 DINO S14+S7 top-100 候选池，不能在 target 切片上扫描手工时序权重。

最终方法的新颖性不应声称来自单个模块，而应定位为：在冻结 DINO 旋转检测器中，将高分辨率补充候选、source-safe 多线索因果关联、native fallback 和后置周期 OBB 稳定统一起来，以同时处理暗光语义排序和小目标空间分辨率问题，并取消 target-scope 路由。

---

## 5. 当前接口与复现入口

<a id="sec-5"></a>

<a id="sec-5-1"></a>

### 5.1 模型与组件身份入口

| 组件 | 路径 | 说明 |
| --- | --- | --- |
| 正式 DINO 组件 | `work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth` | native S14 + ROI classifier `alpha=0.5` + S7 disabled；唯一正式部署基线 |
| DINO 小头正式训练结果 | `work_dirs/dino_teacher_scoped_lowlight_v1_formal8/` 下的 `train_result.json`、`labeller_best_source_only.pth`、`formal_integrated_test/results.pkl` | 8 epoch source-only 训练与完整 test 产物 |
| BrightAug checkpoint | `work_dirs/crane_symeood_k1_brightaug/epoch_20.pth` | 完整 test 命令行传入的、未经修改的 BrightAug checkpoint |
| 普通 SymEOOD K1 | `work_dirs/crane_symeood_k1/epoch_24.pth`，SHA256 `57233e5423de4a9d0a67fd51058cd7d92adaac5d6621f275cc0ec5b0fc7f9ee2` | 普通 K1，不是 V4+seq11-v2 |
| V4+seq11-v2 暂定候选 | `work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407/epoch_10.pth` | 历史暂定 source-retention candidate，`source_gate_passed=False` |
| 联合推理 config | `crane_project/configs/crane_symeood_dino_unified_v1.py` | 见 5.2 与 6.2 |
| 正式 DINO 组件 config | `crane_project/configs/crane_symeood_formal_dino_native_s14_v1.py` | native-S14 纯 DINO 身份，与 SymEOOD K1、融合候选不同 |

<a id="sec-5-2"></a>

### 5.2 实验入口与协议版本索引

| 协议 / 阶段 | 入口 | 用途 |
| --- | --- | --- |
| 暗光 DINO 小头 | `crane_project/tools/dino_teacher_rotated_labeller.py` + `crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py` | source-only 8 epoch 训练与完整 test |
| 远距候选初筛 | `crane_project/tools/dino_teacher_far_distance_candidate_audit.py` | 已被逐级审计取代 |
| 候选覆盖审计 | `dino_teacher_token_scale_rpn_coverage_audit.py`、`dino_teacher_rpn_roi_attrition_latency_audit.py` | 只读归因与延迟基线 |
| 分类器插值 / NMS | `dino_teacher_fc_cls_interpolation_selector.py`、`dino_teacher_rotated_nms_retention_audit.py` | 选定 `alpha=0.5` 与 NMS IoU `0.5` |
| S7 第一版 | `dino_teacher_rotated_labeller.py --s7-residual --train-components s7_rpn` | 第一版 readout/RPN |
| protocol-12 / 13 | `--train-components s7_lane_arbitration` | lane arbitration v1 / v2 |
| protocol-14 / 15 | `--train-components s7_quality_suppression` | quality suppression 与 v15 预检 |
| protocol-16 / 19 | `crane_project/utils/s7_temporal_association.py`、`--s7-temporal-association` | 统一因果时序选择与环境归因审计 |
| 阶段二B | `--s7-temporal-quality-head` | relative-quality 细化 |
| 固定三切片审计 | `dino_teacher_s7_temporal_fixed_target_audit.py`、`dino_teacher_s7_temporal_source_attribution_audit.py`、`dino_teacher_s7_temporal_immediate_override_audit.py` | 严格只读诊断 |
| 阶段三 student | `crane_project/tools/dino_teacher_s7_temporal_student_train.py` + `crane_project/configs/crane_symeood_scoped_dino_lowlight_s7_temporal_student_v1.py` | source-only 候选排序学生 |
| protocol-22 | `s7_candidate_static_head`（同一 labeller 入口） | 静态域泛化 ranker |
| selective promotion V1 | `dino_teacher_s7_selective_promotion_train.py` + `--train-components s7_selective_promotion` | native-protected 成对接管 |
| selective promotion V2（标量版） | `dino_teacher_s7_small_temporal_ranker_train.py` + `--s7-selective-two-frame` | 两帧常速度 + 24 标量 pair head |
| protocol-23 / 24 / 25 | `dino_teacher_s7_highres_roi_ranker_train.py`、`dino_teacher_s7_highres_margin_source_audit.py`、`dino_teacher_s7_highres_fixed_target_audit.py` | stride-7 ROI ranker 与 margin/固定 target 审计 |
| protocol-26 | `dino_teacher_s7_unified_highres_margin_source_audit.py` | unified hard-pair ranker 与两级裁决 |
| protocol-27 / 28 | `--smooth-geometry-ranking` | smooth geometry 支持审计与训练选项 |
| protocol-29 | `dino_teacher_s7_native_relative_risk_v2` | native-relative risk residual |
| protocol-30 / 31 / 32 | `dino_teacher_s7_paired_view_role_switch_support_audit.py`、`dino_teacher_s7_sequence_crossfit_support_audit.py` | 支持度审计 |
| protocol-33 / 34 / 35 | `dino_teacher_s7_dense_temporal_separability_audit.py`、`dino_teacher_s7_roi_temporal_contrastive_train.py` | 对象级时序可分性、contrastive adapter、反事实归因 |
| 论文指标统一入口 | `mmrotate/datasets/crane_custom_dota.py`、`crane_project/tools/eval_crane_offline.py` | `paper_temporal=True` 时输出完整指标 |

<a id="sec-5-3"></a>

### 5.3 统一复现命令

**（a）基础 DINO 小头训练（完整可运行命令）**

```bash
PYTHONPATH=. python3 \
  crane_project/tools/dino_teacher_rotated_labeller.py \
  --data-root crane_project/data/crane_grab/ \
  --source-train-datasets train:train train_sim:train \
  --source-val-datasets val:val \
  --dinov2-repo third_party/dinov2 \
  --dinov2-checkpoint pretrained/dinov2_vitl14_pretrain.pth \
  --dinov2-model dinov2_vitl14 \
  --dino-gpus 1 2 --head-gpu 0 \
  --legacy-sdpa-query-chunk 512 \
  --dino-height 600 --dino-max-long-side 1333 --patch-size 14 \
  --rpn-feat-channels 256 --roi-fc-channels 1024 \
  --roi-samples 256 --proposal-count 2000 --max-detections 2000 \
  --epochs 8 --lr 0.001 --momentum 0.9 --weight-decay 0.0001 \
  --max-grad-norm 10 --warmup-iters 1000 --warmup-ratio 0.001 \
  --lr-steps 5 7 --lr-gamma 0.1 \
  --checkpoint-interval 1 --selection-epochs 1 2 3 4 5 6 7 8 \
  --feature-cache-dir work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache \
  --work-dir work_dirs/dino_teacher_scoped_lowlight_v1_formal8 \
  --seed 0 \
  --out-json work_dirs/dino_teacher_scoped_lowlight_v1_formal8/train_result.json \
  --skip-target-eval
```

`CUDA_VISIBLE_DEVICES=0,1,2` 只用于模型分片，不使用 DDP，也不改变有效 batch size。

**（b）S7 / 时序 / 接管实验族的公共配置与增量**

以下四条命令（S7 第一版、lane arbitration、temporal association、temporal quality head）共享 38 个参数；下表给出各自独有的参数。合并方式已用脚本逐 token 核对：四条原命令的公共项与增量按原顺序重建后与原文完全一致（flag 集合与相对顺序均通过），未改变任何参数取值。完整命令的精确 flag 顺序仍以归档原文为准（见[来源索引](#appendix-sources)）。

公共块（`CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. python3 crane_project/tools/dino_teacher_rotated_labeller.py`）：

```text
--data-root crane_project/data/crane_grab/
--source-train-datasets train:train train_sim:train
--source-val-datasets val:val
--dinov2-repo third_party/dinov2
--dinov2-checkpoint pretrained/dinov2_vitl14_pretrain.pth
--dinov2-model dinov2_vitl14
--dino-gpus 1 2
--head-gpu 0
--legacy-sdpa-query-chunk 512
--dino-height 600
--dino-max-long-side 1333
--patch-size 14
--rpn-feat-channels 256
--roi-fc-channels 1024
--roi-samples 256
--proposal-count 2000
--max-detections 2000
--roi-nms-iou-thr 0.5
--s7-residual
--s7-channels 128
--s7-rpn-feat-channels 128
--s7-proposal-count 500
--s7-nms-pre 2000
--s7-anchor-sizes 16 32 64 128 256
--source-small-repeat 1
--epochs 4
--lr 0.001
--momentum 0.9
--weight-decay 0.0001
--max-grad-norm 10
--warmup-iters 1000
--warmup-ratio 0.001
--lr-steps 2 3
--selection-epochs 1 2 3 4
--checkpoint-interval 1
--feature-cache-dir work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache
--skip-target-eval
--seed 0
```

各实验独有的参数：

| 实验 | 与公共块的差异 |
| --- | --- |
| S7 first implementation（`s7_rpn`） | `--s7-source-min-full-top1 677`、`--s7-source-min-small-top1 303`、`--s7-source-max-mcml 3`、`--train-components s7_rpn`、`--init-checkpoint work_dirs/dino_teacher_fc_cls_interpolation_v1/source_safe_interpolated_head.pth`、`--work-dir work_dirs/dino_teacher_s7_residual_v1`、`--out-json work_dirs/dino_teacher_s7_residual_v1/train_result.json` |
| lane arbitration v1（`s7_lane_arbitration`） | `--s7-merge-init-bias -2.0`、`--s7-lane-hidden 32`、`--s7-lane-max-adjustment 2.0`、`--s7-lane-base-epoch 1`、`--train-components s7_lane_arbitration`、`--init-checkpoint work_dirs/dino_teacher_s7_retention_merge_v1/labeller_epoch_01_source_only.pth`、`--work-dir work_dirs/dino_teacher_s7_lane_arbitration_v1`、`--out-json work_dirs/dino_teacher_s7_lane_arbitration_v1/train_result.json` |
| temporal association v1（`s7_temporal_association`） | `--s7-merge-init-bias -2.0`、`--s7-temporal-association`、`--s7-temporal-base-epoch 1`、`--s7-temporal-margin 0.5`、`--s7-temporal-retention-weight 2.0`、`--s7-temporal-gain-weight 1.0`、`--s7-temporal-prior-weight 0.01`、`--s7-temporal-max-candidates 100`、`--s7-temporal-min-confirmations 2`、`--s7-temporal-override-margin 0.25`、`--s7-temporal-max-center-distance 3.0`、`--s7-temporal-min-riou 0.05`、`--s7-temporal-min-appearance 0.20`、`--s7-source-min-full-top1 688`、`--s7-source-min-small-top1 311`、`--s7-source-max-mcml 3`、`--train-components s7_temporal_association`、`--source-retain-max-top1-drop 0`、`--init-checkpoint work_dirs/dino_teacher_s7_retention_merge_v1/labeller_epoch_01_source_only.pth`、`--work-dir work_dirs/dino_teacher_s7_temporal_association_v1`、`--out-json work_dirs/dino_teacher_s7_temporal_association_v1/train_result.json` |
| quality head（`s7_temporal_association` + quality head） | 同 temporal association，但把 `--s7-temporal-margin/retention-weight/gain-weight/prior-weight` 替换为 `--s7-temporal-quality-head`、`--s7-temporal-quality-hidden 128`、`--s7-temporal-quality-loss-weight 1.0`，并把 `--work-dir`/`--out-json` 改为 `work_dirs/dino_teacher_s7_temporal_quality_association_v1` |

**（c）正式完整 test**

```bash
PYTHONPATH=. python3 tools/test.py \
  crane_project/configs/crane_symeood_scoped_dino_lowlight_v1.py \
  work_dirs/crane_symeood_k1_brightaug/epoch_20.pth \
  --gpu-ids 0 \
  --out work_dirs/dino_teacher_scoped_lowlight_v1_formal8/formal_integrated_test/results.pkl \
  --eval mAP \
  --work-dir work_dirs/dino_teacher_scoped_lowlight_v1_formal8/formal_integrated_test
```

命令行 checkpoint 仍是未经修改的 BrightAug checkpoint；DINO 小头 checkpoint、DINOv2 权重、scope 和稳定器系数均由 config 固定加载。target-dev 暗光诊断结果作为方法形成过程的历史证据保留在文档和结果文件中；正式组合推理与完整 test 统一使用本条命令，不再维护另一套离线结果拼接脚本。

**（d）服务器 `work_dirs` 清理边界**

`feature_cache/` 不属于论文证据本身。清理时必须保留正式 baseline、阶段二 source-gated checkpoint、阶段三 target-dev 结果、static ranker 的 source-only 结果和 selective promotion V1 的 `train_result.json`，且不得删除下列目录（除非已另行迁移其 checkpoint 与结果 JSON）：

```text
dino_teacher_scoped_lowlight_v1_formal8/
dino_teacher_fc_cls_interpolation_v1/
dino_teacher_s7_temporal_relative_quality_v1/
dino_teacher_s7_temporal_student_v1/
dino_teacher_s7_temporal_student_target_v1/
dino_teacher_s7_static_domain_ranker_v1/
dino_teacher_s7_selective_promotion_v2/
```

可清理的是以下已关闭路线。先预览、确认结果 JSON 已迁移后再删除，不要凭截图排序直接删：

```bash
cd /home/omnisky/workspace/symEOOD

obsolete_dirs=(
  dino_teacher_s7_quality_suppression_v1
  dino_teacher_s7_lane_arbitration_v1
  dino_teacher_s7_lane_arbitration_v2
  dino_teacher_s7_retention_merge_v1
  dino_teacher_s7_residual_v1
  dino_teacher_roi_cls_pairwise_v2_formal4_v1
)

# 预览：查看体积与顶层文件
for name in "${obsolete_dirs[@]}"; do
  path="work_dirs/$name"
  if [ -d "$path" ]; then
    du -sh "$path"
    find "$path" -maxdepth 1 -type f -print
  fi
done

# 删除（确认已迁移后执行）
for name in "${obsolete_dirs[@]}"; do
  path="work_dirs/$name"
  if [ -d "$path" ]; then
    rm -rf -- "$path"
  fi
done
```

attribution、immediate-override、association 和 quality-attribution 目录暂不纳入自动删除，除非先确认其结果 JSON 已单独保存；它们可能仍是递归时序审计的原始证据。也可只清理已关闭目录中的 `feature_cache/`，保留 JSON、日志和 checkpoint。

<a id="sec-5-4"></a>

### 5.4 资源约定

- 后续单卡命令优先使用物理 GPU0：`CUDA_VISIBLE_DEVICES=0 ... --gpu 0`；多卡训练使用前三张卡：`CUDA_VISIBLE_DEVICES=0,1,2`。
- 每次训练或真实栈 smoke 记录 peak allocated、peak reserved、单帧/step 延迟和实际 forward count。
- 不预先声称 refiner 不增加显存；FPN 特征生命周期延长会增加峰值显存。
- 已知基线：因果历史 refiner smoke 单卡 GTX1080 峰值 allocated 约 `1.11 GiB`、reserved 约 `1.22 GiB`；V4 replay 训练单 GPU0 峰值 allocated `2,348,764,160 B`（约 2.19 GiB）、reserved `2,512,388,096 B`（约 2.34 GiB）。

<a id="sec-5-5"></a>

### 5.5 本轮外部结果文件索引

部分文件只存在于当时的 `/Users/mac/Downloads` 或 Codex attachment 临时目录；迁移到新机器/新账号时需另行复制。

```text
# V4、source inventory 与 measurement-validity
source_inventory_support.json
source_inventory_all_lane_audit.json
fixed_target_causal_history_diagnostic.json

# 单帧 geometry refiner
20260828_092559.log.json
20260828_135345.log.json
source_val_epoch10_results.pkl
source_val_epoch11_results.pkl
geometry_refiner_frozen_contract.json
cuda_peak_memory_rank0.json

# Dual-tower
dual_tower_v2_package.json
source_val_dual_tower_v2_results.pkl
source_val_dual_tower_v2_audit.json
dual_tower_v21_source_val_review.tar.gz
dual_tower_v21_fixed_test_results.pkl

# Causal V2/V3
source_val_causal_history_support_audit.json
20260829_142925.log.json
v2_source_val_review.tar.gz
checkpoint_eval_summary.json
epoch10_source_promotion.json
fixed_test_audit.json
fixed_test_results.pkl
v3_source_val_review.tar.gz
v3_epoch9_fixed_test_review.tar.gz
v3_component_fixed_test_review.tar.gz
source_val_anchor_fallback_attribution.json
fixed_target_anchor_fallback_attribution.json

# seq11 E1 与 seq11-v2
v3_vs_seq11_e1_epoch10_fixed_target_pair.tar.gz
v4_seq11_v2_source_review.tar.gz
```

其中 `checkpoint_eval_summary.json` 和 `fixed_test_results.pkl` 在不同 work dir 中会重名；必须和所属 config、checkpoint SHA256、工作目录一起读取，不能只凭文件名归因。

最新 seq11-v2 复核包：

```text
/Users/mac/Downloads/v4_seq11_v2_source_review.tar.gz
SHA256 = d69484eb766d6e2f0f1fe59ec5f0c70623fd925d6f4a015ebe712dce61d85506
```

包内包含 10 个 epoch 在官方 738 帧和 seq11 aux-val 48 帧上的 `results.pkl`、`checkpoint_eval_summary.json`、训练日志、冻结合同、GPU 记录和 checkpoint SHA256；不包含大 checkpoint，因此压缩包较小是正常的。

<a id="sec-5-6"></a>

### 5.6 迁移与读取顺序

新账号不要只依赖聊天历史。按以下顺序读取即可恢复当前项目状态：

1. 本文档第 1 节（当前结论与模型身份）与第 3 节（实验演进与裁决）。
2. [OBB 观测可靠性与连续输出](OBB观测可靠性与连续输出.md)：小论文当前的检测前端身份与观测接口。
3. [Webots 单目深度估计](Webots单目深度估计.md)：深度接口契约与当前边界。
4. `.workbuddy/memory/MEMORY.md`：项目级历史索引；再按需读取同目录日期文件。

`.workbuddy/memory` 位于项目目录，可随整个工作区迁移。另需注意本仓库 `.gitignore` 会忽略 `docs/*.md` 与 `.workbuddy/memory/*.md`；如果仅通过 Git 迁移，必须显式 force-add 这些文档，或单独复制 `docs/` 与 `.workbuddy/memory/`。

恢复时必须先回答四个问题：

- 当前讨论的是 source-val、seq11 同视频 OOF，还是 fixed TEST？
- checkpoint 是否已经 source-gated，还是仅为训练/暂定候选？
- 指标中的 K1 是 source-val 还是 TEST 口径？
- 是否把抓料阶段 raw 指标和 measurement-valid 指标分开报告？

**项目内记忆文件**

```text
.workbuddy/memory/MEMORY.md
.workbuddy/memory/2026-07-07.md  …  2026-07-28.md（日期日志）
```

**项目外入口（属于原 macOS 账号，新账号不会自动拥有，必须显式复制）**

```text
背景材料
  /Users/mac/Desktop/大论文研究内容/背景.md
  .hermes/desktop-attachments/背景.md

Codex 记忆索引与相关工作摘要
  /Users/mac/.codex/memories/MEMORY.md
  /Users/mac/.codex/memories/memory_summary.md
  /Users/mac/.codex/memories/raw_memories.md
  /Users/mac/.codex/memories/extensions/ad_hoc/notes/2026-06-20-gate-norm-tta-result.md
  /Users/mac/.codex/memories/extensions/ad_hoc/notes/2026-07-26-symeood-dino-innovation.md
  /Users/mac/.codex/memories/extensions/ad_hoc/notes/20260722-185612-shared-filter-oracle-next.md
  /Users/mac/.codex/memories/extensions/ad_hoc/notes/20260723-145450-p3-feature-shift-supported.md
  /Users/mac/.codex/memories/extensions/ad_hoc/notes/20260724-100104-multimodal-fpn-closed-dino-teacher-next.md
  /Users/mac/.codex/memories/extensions/ad_hoc/notes/20260801-symeood-s7-current-handoff.md
  /Users/mac/.codex/memories/skills/symeood-mcml-diagnosis/SKILL.md
  /Users/mac/.codex/memories/rollout_summaries/2026-07-22T11-40-23-2zLw-symeood_s7_lidere_retention_merge_handoff.md
  /Users/mac/.codex/memories/rollout_summaries/2026-07-27T02-59-06-b3wr-symood_dino_unified_small_object_diagnosis_literature_2024_2.md
```

其余 rollout summaries 是更早实验的会话级证据，由全局 `MEMORY.md` 按任务索引；迁移时若需要完整历史，直接复制整个 `/Users/mac/.codex/memories/`（排除其中 `.git/` 即可）。

---

## 6. 未解决问题与适用边界

<a id="sec-6"></a>

<a id="sec-6-1"></a>

### 6.1 小目标最终排序（当前核心未解问题）

`seq03_small` 所代表的未知小目标正确候选虽已进入 top-100，却尚不能稳定排到第一；而既有 source split 又不足以训练并留出验证一个跨 real/sim 的安全接管器。完整的问题定义、支持度证据和停止条件见 [3.2.6](#sec-3-2-6)，行动顺序见 [3.2.7](#sec-3-2-7)。

<a id="sec-6-2"></a>

### 6.2 SymEOOD–DINO 联合前端（已实现，未验证）

<a id="sec-6-2-1"></a>

#### 6.2.1 联合结构的定位

此前最新 native S14 `alpha=0.5` checkpoint 主要通过 labeller 与独立审计入口运行；单独将其包装成纯 DINO config 也不能代表论文中的 SymEOOD 主方法。标准 MMRotate 联合配置为：

```text
crane_project/configs/crane_symeood_dino_unified_v1.py
```

融合方式（以配置正文为准）：SymEOOD 主头先产生一个 SymEOOD top-1 OBB；该 OBB 被转换到 DINO 输入尺度，作为额外 proposal 与 frozen DINO native-S14 RPN proposals 合并；所有候选再统一经过同一个 source-safe DINO ROI 分类/回归头，最终只输出一个 top-1 OBB。配置明确 `raw_cross_model_score_comparison=False`，即 SymEOOD 原始 score 不参与融合；`source_owned_geometry=True`，SymEOOD 候选保留原始几何、DINO 候选保留 DINO ROI regressed 几何。这既不是纯 DINO，也不是两个检测结果的启发式二选一。配置另明确 `scope_manifest=None`、`scope_policy='all_frames'`、`target_scope=False`、`sequence_identity_routing=False`、`s7_enabled=False`、`s7_temporal_takeover=False`、`stabilizer.enabled=False`（`alpha=1.0`），即 BrightAug、target scope、序列身份路由、S7、时序接管和框稳定器全部关闭。

**配置核查结果与一处待核验冲突。** 本轮直接读取了该配置文件（139 行），发现与归档记录（来源 D 第 10.18.22 节，2026-08-08）在 SymEOOD 位置权重上不一致：

| 项目 | 归档记录（2026-08-08） | 当前配置文件（本轮读取） |
| --- | --- | --- |
| 基座 config | 「保留正式 `crane_symeood_k1.py`」 | `_base_ = ['./crane_symeood_k1_brightaug.py']`；`sym_eood_config = crane_project/configs/crane_symeood_k1_brightaug.py` |
| SymEOOD 位置权重 | `work_dirs/crane_symeood_k1/best_Weighted_R_center_epoch_12.pth` | `work_dirs/crane_symeood_k1_brightaug/epoch_20.pth` |
| DINO 头权重 | `source_safe_interpolated_head.pth`（一致） | 同左 |
| work_dir | 未记录 | `work_dirs/crane_symeood_dino_unified_v2/full_test`（文件名为 `unified_v1`） |
| 记录依据 | 「原始 SymEOOD 日志按 source-val `Weighted_R_center` 选择 epoch12=`0.9945`，高于 epoch24=`0.9932`」 | 配置未记录该选择依据 |

两者不能同时成立，本轮**不做裁决**：可能是配置在记录之后被改写，也可能记录描述的是当时的另一个变体。可核实的旁证是，本工作区同时存在 `work_dirs/crane_symeood_k1/best_Weighted_R_center_epoch_12.pth` 与 `work_dirs/crane_symeood_k1/epoch_24.pth`，而 `work_dirs/crane_symeood_k1_brightaug/` 目录为空、`work_dirs/*unified*` 不存在。因此该联合前端在本工作区既没有可用的 BrightAug 位置权重，也没有任何运行产物。

DINO checkpoint provenance contract 仍要求 selector/protocol/alpha、target-unread、source gate、source full/small 摘要和 S7-disabled 状态一致（`alpha=0.5`、`require_target_unread=True`、`require_source_gate=True`、`require_s7_disabled=True`、`source_full=677/3`、`source_small=303/3`）。

该工作形成了真正的联合推理结构，但仍是一个**新的、尚未验证的组合模型**：`formal_detection_contract` 中 `joint_source_gate_required=True` 且 `detector_training_required=False`，即联合 source gate 尚未运行。此前 SymEOOD 与 DINO 的各自数字只能作为组件证据，不能相加或冒充联合模型结果。当前没有重新读取固定三片段 target-dev；必须先运行联合 source gate，确认 old-correct retention、full/small Top-1、MCML、DFR/ACI 与仿真角度指标不退化，之后才决定是否授权完整 test。后续对象级深度 head 应只接收该统一前端冻结后的 Top-1 OBB、score 及可选 ROI feature，不得反向改变检测排序。

<a id="sec-6-2-2"></a>

#### 6.2.2 几何融合路线与 seq11 补充数据

这条支线的目标是判断「SymEOOD 主头 + 冻结 DINO 候选」能否形成统一前端。所有路线都在同一前端身份下评估；`source-val` 与 `fixed TEST` 数字来自不同 split，不能直接混合比较。

**基线指标与容易混淆的口径**

K1 官方 source-val（738 帧）：

| 域 | R_center | mean RIoU | DFR | ACI | A-RMSE | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 98.67% | 0.7851 | 3.4797 | 0.9396 | — | 1 |
| sim | 99.02% | 0.8891 | 2.0363 | 0.9565 | 17.8578° | 2 |

K1 fixed TEST（用户正式记录）：

| 域 | R_center | mean RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 99.27% | 0.8046 | 2.5786 | 0.9393 | — | 80.85% | 44 |
| sim | 100% | 0.9087 | 2.0925 | 0.9576 | 1.6520° | 100% | 0 |

之前出现的「K1 指标不一致」就是因为把 source-val 与 TEST 记录混在了一起。

**路线 A：固定 V4 输出级路由 + 抓料阶段审计**

```text
real：DINO 有框则采用 DINO native OBB，否则回退 SymEOOD
sim：采用 SymEOOD
```

该 V4 是固定输出级组合，不是联合训练模型，也不是最终推荐结构。输入为 992 帧 all-lane JSON，输出来源 `dino_native=420`、`sym_eood=572`：

| 域 | R_center | mean RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 95.95% | 0.6789 | 7.7873 | 0.8587 | — | 100% | 6 |
| sim | 100% | 0.8857 | 1.8932 | 0.9621 | 1.5487° | 100% | 0 |

- 相对 SymEOOD：`gained=162`、`lost=5`。
- 原 gate 只因 `real MCML=6 > 5` 失败。
- 两通道 oracle 在 `real_seq03` 的最长失败仍是 `140–145` 共 6 帧，说明这 6 帧并非再调两分支选择规则就能解决。
- 虽然用户允许工程门槛放宽为 6，但 V4 的 real RIoU、DFR、ACI 明显下降，不能仅凭 MCML 接近门槛把 V4 当最终模型。

抓料阶段 measurement-validity：操作规则把 `real_seq03` `130–187`（共 58 帧）统一标为 `material_contact_grabbing_or_not_fully_detached`；这 58 帧中有 48 个 hit、10 个 miss，不能只删除失败帧。排除后的 measurement-valid real 指标为 frame `362/420`（有效比例 `86.19%`）、R_center `96.69%`、mean RIoU `0.6958`、DFR `7.5527`、ACI `0.8562`、MCML_max `4`。结论：measurement-valid gate 通过，但原始 gate 仍失败；排除抓料阶段没有解决 V4 的几何与时序稳定性问题。论文可以并列报告 raw 与 measurement-valid，并说明抓料阶段由作业人员控制，但不能用该排除代替模型改进。

主要服务器产物：

```text
work_dirs/crane_symeood_dino_application_domain_v4/fixed_target_json_audit.json
work_dirs/crane_symeood_dino_conservative_takeover_v2/full_test_metric_v2/conservative_takeover_fusion_audit.json
```

**路线 B：source-val 几何可修正性审计**（工具 `crane_project/tools/symeood_dino_geometry_refinability_audit.py`）

738 帧 source-val 关键结果：

- DINO presence：`737/738=99.86%`；real 为 `226/226=100%`。
- real GT 中心在 DINO 框内：`100%`；real 中心误差 median `5.60 px`、p90 `12.58 px`。
- real both-present 223 帧：SymEOOD RIoU `0.7957`，DINO `0.6667`。
- DINO + Sym 尺寸：`0.7717`；DINO + Sym 角度：`0.6650`。
- DINO 中心 + GT 几何的 real oracle：`0.8578`。
- candidate oracle：real RIoU `0.7982`、DFR `4.0537`、ACI `0.9260`、MCML `0`；sim RIoU `0.8927`、DFR `2.1214`、ACI `0.9555`、MCML `0`。

有效结论：DINO 能提供存在性与中心支撑，主要几何问题是尺度，尤其是宽度。该审计授权训练 source-only refiner，不授权 TEST、部署，也不支持继续训练 takeover router。

**路线 C：单帧 Size-only / Full geometry refiner**

实现目的：DINO 保持目标发现能力；从冻结 SymEOOD FPN 对 DINO proposal 做 Rotated ROIAlign；预测局部中心、log 尺寸和双角度残差；trainer 与 runtime 共用同一个 refiner、coder、active-component mask 和 decode；real/sim 同一 forward；DINO missing 统一回退 SymEOOD；不使用域路由。

```text
mmrotate/models/roi_heads/dino_conditioned_geometry_refiner.py
mmrotate/models/detectors/symeood_dino_geometry_refiner_trainer.py
mmrotate/datasets/pipelines/loading.py
mmrotate/core/hooks/geometry_refiner_contract_hook.py
crane_project/tools/symeood_dino_geometry_refiner_source_smoke.py
crane_project/tools/symeood_feature_reuse_equivalence.py
crane_project/configs/crane_symeood_dino_geometry_refiner_size_source_v1.py
crane_project/configs/crane_symeood_dino_geometry_refiner_full_source_v1.py
```

已锁定的技术合同：

- 中心残差使用 DINO 局部归一化坐标；
- `le90 + edge_swap + proj_xy`，处理宽高交换等价框；
- 角度零初始化使用 `sin(2Δθ)` 与 `cos(2Δθ)-1`，零输出严格等于 DINO；
- DINO 原图框在增强前注册为 rotated bbox field，和 GT 同步 resize/flip；
- 冻结 baseline 参数与 BN buffer，训练前后哈希一致；
- `simple_test_from_features()` 复用一次 FPN，关闭 refiner 时必须与旧 `simple_test()` 逐元素等价。

结果与裁决：Size-only 的训练与冻结合同通过，但无法同时恢复 RIoU、DFR、ACI，未成为正式候选；Full 有部分 R_center/ACI 改善，但 DFR 与 sim A-RMSE 明显不理想，未通过最终 source gate。结论：几何容量审计成立，但一个单帧 ROI 残差头没有把 oracle 容量转化为可泛化性能。本轮外部结果文件：`20260828_092559.log.json`、`source_val_epoch10_results.pkl`、`source_val_epoch11_results.pkl`、`geometry_refiner_frozen_contract.json`。

**路线 D：Dual-tower V2 / V2.1**

设计：把 Size-only 与 Full 的 size/pose 分支组合成双塔；V2.1 继续训练 size 分支，希望在保留中心/角度的同时改善尺度与 DFR。

```text
crane_project/configs/crane_symeood_dino_geometry_refiner_dual_tower_source_val_v2.py
crane_project/configs/crane_symeood_dino_geometry_refiner_dual_tower_size_source_v21.py
crane_project/configs/crane_symeood_dino_geometry_refiner_dual_tower_v21_fixed_test.py
crane_project/tools/symeood_dino_dual_tower_v2_audit.py
crane_project/tools/symeood_dino_dual_tower_v2_package.py
crane_project/tools/symeood_dino_dual_tower_v21_source_gate.py
crane_project/tools/symeood_dino_dual_tower_v21_promote.py
```

V2.1 source-val 最好候选为 epoch7，composite 最高 `0.0093115`：real RIoU `0.7521`、DFR `4.4628`、ACI `0.9253`；sim RIoU `0.8052`、DFR `4.6079`、ACI `0.9075`。虽通过当时的宽松 composite gate，但改善量很小，绝对几何/时序指标仍差；fixed TEST 进一步出现 `MCML=22` 和较高 DFR，因此关闭为最终路线。

**路线 E：因果历史支持审计与历史 refiner**

source-val 支持审计：real 当前 miss 11 帧，1–4 帧历史 own-hit 支持率均为 100%，直接 hold 支持 `6/11=54.5%`；real 常速度支持 `5/11=45.5%`；sim 当前 miss 2 帧，2–4 帧历史可找到 own-hit，但简单 hold 为 0。

fixed-target 支持审计：real 当前 miss 26 帧，历史窗口 1→4 时 own-hit 支持率从 `65.4%` 到 `88.5%`；4 帧历史 oracle 可把 MCML `6→3`，但它是 GT oracle，不是部署结果；常速度只支持 `5/26=19.2%`，不能降低 MCML。

```text
crane_project/tools/symeood_dino_causal_history_support_audit.py
crane_project/configs/crane_symeood_dino_causal_history_refiner_source_v1.py
crane_project/tools/symeood_dino_causal_history_source_smoke.py
crane_project/tools/symeood_dino_causal_history_source_gate.py
```

历史 refiner 采用因果历史 ROI、当前帧 anchor、无 sequence/frame 身份输入、无域路由；无有效历史时严格退化为 current-only。正式 smoke 使用普通 K1 `work_dirs/crane_symeood_k1/epoch_24.pth`（而不是 BrightAug `epoch_20.pth`），smoke 最终通过，单卡 GTX1080 峰值 allocated 约 1.11 GiB、reserved 约 1.22 GiB。

**路线 F：K1-anchored causal V2 与 retentive V3**

V2 fixed TEST：

| 域 | R_center | RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 95.24% | 0.7001 | 5.9190 | 0.9051 | — | 97.46% | 19 |
| sim | 100% | 0.8908 | 1.9488 | 0.9603 | 1.5378° | 100% | 0 |

sim 保持较好，但 real MCML=19、DFR 高、几何下降，失败。

V3 source gate：V3 引入 K1 几何 anchor、bounded residual、continuous retention 和相邻帧误差一致性。10 个 epoch 中仅 epoch9 通过当时的完整 source gate：K1→V3 的 real RIoU `0.7690→0.7758`、sim RIoU `0.8838→0.8909`、sim A-RMSE `9.0415°→5.8378°`、sim MCML `2→0`。该结果只允许 source promotion，不代表 fixed TEST、未知序列或部署。

V3 fixed TEST 与四组件消融：epoch9 fixed TEST 没有复现 source-val 的改善，real DFR 仍差；full/current-only/center-only/K1-identity 四模式在 TEST 上只有约 `0.00x` 的浮动，不能称为有意义改进；anchor/fallback attribution 表明这是结构性问题，而不是某个历史分量的单独开关问题。结论：历史信息不是天然免疫域差——历史网络仍会学习 source 的运动、外观、相机抖动和失效模式，训练/TEST 视频差异会导致 history gate 和残差预测失配。

**路线 G / H：seq11 补充数据（59 帧 pilot 与 251 帧 v2 replay）**

seq11 初始 59 帧 pilot 的数据清理：磁盘曾显示 118 图像和 118 标注，但其中 59 个是 macOS AppleDouble `._*` sidecar；真实样本是 manifest 中的 59 帧，不是 118 帧。初始 source inventory：frames `59`、SymEOOD hit rate `0.4915`、DINO hit rate `0.9322`、router support 不满足、证据用途 `RETENTION_AND_NONREGRESSION_ONLY`。旧 48/11 block split 的辅助 val 11 帧为 K1 hit 4、K1 missing + DINO hit 7、K1 present-wrong + DINO hit 0，因此 `target_support=0`，裁决 `STOP_SEQ11_AUX_SUPPORT_INSUFFICIENT`：这不是路径错误，而是该 11 帧不支持「有框但框错」的接管监督。

59 帧加入训练后的 E1 fixed-target 诊断：

| 域 | R_center | RIoU | DFR | ACI | A-RMSE | TDR | MCML_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 93.57% | 0.6763 | 5.8061 | 0.8951 | — | 98.98% | 13 |
| sim | 100% | 0.8753 | 2.0235 | 0.9579 | 1.6343° | 100% | 0 |

新增困难数据有一定作用，但数量太少、real 几何与 DFR 仍差，不能作为最终结果；同视频数据也不提供未知序列证据。

seq11-v2 扩至 251 帧。数据事实：服务器目录 `crane_project/data/crane_grab/extra_source_real_seq11_pilot_k1p9_v2`；看到的 502 图像/JSON/TXT 文件 = 251 个真实样本 + 251 个 `._*` sidecar；有效 251 帧均采用 `k0=1.9`、top-beam-only 标注；manifest 划分 train 203 / aux-val 48，文件 overlap 0；物化目录使用 hardlink，不把同一帧同时放进 train/val/test。

数据/审计产物：

```text
crane_project/data/crane_grab/extra_source_real_seq11_pilot_k1p9_v2/split_manifest.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/full_source_contract.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/blocksplit_v2/audited_split_materialization.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/blocksplit_v2/train_all_lane_audit.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/blocksplit_v2/aux_val_all_lane_audit.json
```

V4 replay 训练设计：

- Base V3 epoch9 初始化并作为 teacher，retention loss weight `0.25`；
- 原始 source 2781 帧 replay + seq11 train 203 帧，总 `2984` 帧；
- 采样节奏 original:aux = `14:1`，防止单个视频主导训练；
- 固定每 epoch 1391 optimizer steps，共 10 epochs、13910 steps；
- seq11 使用相邻帧监督；不把 sequence/frame ID 输入模型；real/sim 共用同一 forward；无 domain routing；
- baseline/teacher 冻结，训练前后哈希一致；
- 单 GPU0；GTX1080 peak allocated `2,348,764,160 B`（约 2.19 GiB），reserved `2,512,388,096 B`（约 2.34 GiB）。

```text
crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay.py
crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_aux_val.py
crane_project/configs/crane_symeood_k1_seq11_v2_aux_val_eval.py
work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407
```

训练和 official source-val 结果：smoke 输出 `ALLOW_K1_RETENTIVE_SEQ11_V2_REPLAY_SOURCE_TRAINING`；full coverage 显示原 2781 与 aux 203 在 10 epoch 中均被覆盖；学习率为 base `5e-5`，epoch1 warmup，2–6 约 `5e-5`，7–9 降至约 `1e-5`，epoch10 约 `5e-7`（日志显示为 `0.0` 是格式化，不是学习率真的为 0）；epoch10 为暂定多目标折中候选。逐 epoch 官方 738 帧指标：

| epoch | real RIoU | real DFR↓ | real ACI↑ | real MCML↓ | sim RIoU | sim DFR↓ | sim ACI↑ | sim A-RMSE↓ | sim MCML↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7753 | 2.7960 | 0.9414 | 0 | 0.8884 | 2.2134 | 0.9551 | 7.0519 | 0 |
| 2 | **0.7895** | 2.7478 | 0.9409 | 0 | **0.8971** | 2.1785 | 0.9550 | 7.0549 | 0 |
| 3 | 0.7482 | 2.7885 | 0.9408 | 0 | 0.8475 | 2.1526 | 0.9553 | 9.8489 | 1 |
| 4 | 0.7853 | 2.7122 | 0.9418 | 0 | 0.8790 | 2.1809 | 0.9562 | 7.1324 | 0 |
| 5 | 0.7874 | 2.6130 | **0.9422** | 0 | 0.8708 | 2.1652 | 0.9564 | 7.0962 | 0 |
| 6 | 0.7682 | **2.5174** | 0.9412 | 0 | 0.8638 | 2.1626 | 0.9561 | 5.8379 | 0 |
| 7 | 0.7756 | 2.5925 | 0.9417 | 0 | 0.8818 | 2.1831 | 0.9563 | **5.7905** | 0 |
| 8 | 0.7760 | 2.6302 | 0.9416 | 0 | 0.8877 | 2.1703 | 0.9565 | 7.0225 | 0 |
| 9 | 0.7806 | 2.6664 | 0.9418 | 0 | 0.8917 | **2.1496** | **0.9566** | 7.0264 | 0 |
| 10 | 0.7828 | 2.6650 | 0.9418 | 0 | 0.8937 | 2.1639 | **0.9566** | 5.7957 | 0 |

所有 epoch 的 `real/R_center=100%`、`real/TDR_w10=100%`；除 epoch3 外 sim MCML 也为 0。

**epoch10 是否考虑了 DFR**：考虑了，但必须准确表述——DFR 是**稳定性约束和候选间次级比较项**；epoch10 不是 real DFR 最优，也不是 sim DFR 最优；epoch10 被暂定保留，是因为它同时满足当前最关键的多指标组合：两域 `MCML=0`，real/sim RIoU 均高于 Base V3，且 sim A-RMSE 略优于 Base V3；当前打包产物没有包含可直接重算的 Base V3 DFR 对照，因此不能仅凭该压缩包宣称「epoch10 的 DFR 相对 Base V3 已正式非退化」。

Base V3 epoch9 的已验证 source-val 对照为 real RIoU `0.7758`、sim RIoU `0.8909`、sim A-RMSE `5.8378°`、real/sim MCML 均为 `0`。对比：epoch6 的 real DFR 最好，但 real RIoU `0.7682`、sim RIoU `0.8638`，几何损失明显；epoch9 的 sim DFR 最好且 RIoU 也通过，但 A-RMSE `7.0264°` 明显变差；epoch2 两域 RIoU 最好，但 A-RMSE `7.0549°` 明显变差；epoch7 的 A-RMSE 最好，但 sim RIoU `0.8818` 低于 Base V3；epoch10 的 real RIoU `+0.0070`、sim RIoU `+0.0028`、A-RMSE `-0.0421°`，两域 MCML 保持 0，DFR 虽非最优但没有出现前期双塔方案的 4–8 级别恶化。因此 epoch10 是**多目标折中候选**，正式表述应为：

> 在官方 738 帧 source-val 上，epoch10 是当前同时满足 Base V3 几何/角度非退化与 MCML 保持的暂定候选；DFR 被作为稳定性约束考虑，但未被单独最小化。

**当前关键缺陷：48 帧辅助验证不覆盖困难样本**。251 帧全量类别为 K1 hit `205`、K1 missing/wrong 且 DINO hit `32`、两者都 bad `14`。现有划分：train 203（K1 hit 157、hard 32、both-bad 14）、aux-val 48（K1 hit 48、hard 0、both-bad 0）。因此：

- aux-val 48 上 K1 和 V4 各 epoch 都接近/达到 mAP 1.0，无法区分模型；
- 文件 overlap=0，所以不是直接帧泄漏；
- 但困难模式全部进入训练，形成**困难支持分配失败/选择盲区**；
- 不能用该 aux-val 为 epoch10 提供「seq11 困难机制有效」的证据。

<a id="sec-6-2-3"></a>

#### 6.2.3 几何融合路线总裁决

本表每一行的「状态」指**该路线在其记录日期内的裁决**，不是全项目的当前结论；各行分属不同模型身份与协议层，不能合并成同一个「最终模型」判断。

| 路线（记录日期） | 该路线状态 | 可保留的结论 |
| --- | --- | --- |
| V4 real-DINO / sim-Sym 固定路由 | 失败为最终模型 | MCML 接近门槛，但 real 几何与时序明显下降 |
| 抓料阶段 measurement-validity | 有效的评估协议 | raw 与有效区间应双报告；不能代替模型优化 |
| takeover / router | 关闭 | source 正样本支持不足，不能安全学习接管 |
| geometry capacity audit | 有效 | DINO 存在性/中心可靠，主要误差为尺寸 |
| Size-only / Full refiner | 失败 | 可训练合同成立，但未把 oracle 容量转化为性能 |
| dual-tower V2/V2.1 | 失败 | source 宽松 gate 的小增益未在 TEST 成立，MCML=22 |
| 简单 hold / velocity | 仅诊断 | 历史有一定 support，但简单外推覆盖不足 |
| causal V2 | 失败 | sim 好、real MCML19/DFR 高 |
| retentive V3 | source 有效、TEST 不稳 | epoch9 是 source promotion-only，域差仍明显 |
| seq11 初始 59 帧 E1 | 弱正向诊断 | 数据有用迹象，但量少且 MCML13 |
| seq11-v2 251 帧 V4 replay | 当时（2026-09-05）本支线内最值得继续 | official 738 上 epoch10 多指标小幅正向，但 aux-val 困难支持为 0；仅指检测融合 seq11 支线，不是全项目当前结论 |
| `crane_symeood_dino_unified_v1` 联合前端 | 已实现、未验证 | 结构成立；必须先通过联合 source gate |

**未执行的条件性后续（seq11 支线）**

该计划尚未被后续裁决取代，也未执行。前置条件：先保留、不再修改 `epoch_10.pth`，并且在对 251 帧完成三折时序块 OOF/CV 之前，不称其为正式最优、不生成 `source_gate_passed=True` 的 promotion artifact、不根据 fixed TEST 调 V4 参数、不宣称未知序列泛化或部署。

目的不是增加随机拆帧，而是让全部困难帧至少一次处于未见验证块：以连续时序块/困难事件为基本单元，不随机拆相邻帧；三折中每折用约 2/3 seq11 块训练，剩余块验证；32 个 `K1 missing/wrong + DINO hit` 和 14 个 `both-bad` 必须在 OOF 汇总中各出现一次；每折仍 replay 原始 2781 帧并保留 Base V3 teacher；官方 738 帧只做 non-regression gate；OOF 结果要报告困难帧 rescue、old-correct lost、RIoU、DFR、ACI、MCML，而不是只看 mAP。

仓库已有可复用的 block-CV 工具，不应再新造一套：

```text
crane_project/data_contracts/real_seq11_pilot_k1p9_three_window_block_cv_v1.json
crane_project/tools/symeood_dino_seq11_block_cv_materialize.py
crane_project/tools/symeood_dino_seq11_block_cv_gate.py
crane_project/tools/symeood_dino_seq11_block_cv_select.py
crane_project/tools/symeood_dino_seq11_block_cv_support_audit.py
crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v3_seq11_blockcv.py
```

这些旧工具需要先扩展到 251 帧 v2 manifest 与 V4 replay 合同，不能原样把旧 59 帧 contract 当新数据使用。OOF 通过后的顺序为：生成 V4 专用 dual-source gate（明确 official-738 retention 与 seq11 OOF hard-support 两部分）→ 固定 epoch/模型，不再读取 TEST 做选择 → 生成带 checkpoint SHA256、数据合同和 `source_gate_passed=True` 的 promotion artifact → 再运行 fixed TEST 基准，并同时报告 raw 全帧指标、按预先定义的抓料阶段排除后的 measurement-valid 指标，以及抓料阶段被排除的总帧数、hit/miss 数和理由。fixed TEST 只用于报告这个已固定模型，不能反向扫描阈值或继续选 epoch。若三折 OOF 仍不能在困难块上稳定 rescue，当前结论应收束为：seq11 数据揭示了结构性 failure，但同视频补充数据不足以形成可泛化融合模型；需要新的独立真实视频，而不是继续在同一 TEST 上堆规则。

<a id="sec-6-3"></a>

### 6.3 已修复的实现问题（工程修复，非性能证据）

- `tools/test.py` config 缺少 `data.val`。
- `LoadDinoProposalFromAudit` 强制 all-lane 完整性。
- optimizer 把 `momentum` 传给 AdamW。
- `data.samples_per_gpu` 与 `data.train_dataloader.samples_per_gpu` 重复。
- frozen hash 在公开初始化前后取值时机不一致。
- `simple_test` 中 list/tensor 索引类型错误。
- `flip` 元数据缺失；增加显式 no-flip metadata。
- history `DataContainer.pad_dims` 触发 collate assertion。
- 误用 BrightAug `epoch_20`，改回普通 K1 `epoch_24`。
- dual-tower 推理时「至少一支可训练」的构造约束错误。
- child config 对 base `None` dict 合并时缺少 `_delete_=True`。
- seq11 `._*` AppleDouble 被误计为样本。
- split 名称的字符串前缀安全检查过严，改为规范化 data-root 子路径检查。
- Hook 的 `priority` 被写进 Hook 实例，触发 MMCV 保留属性错误。
- config 顶层保留打开的 JSON 文件句柄，导致 MMCV deepcopy `TextIOWrapper` 失败；改为局部读取并关闭。
- fixed target audit 的嵌套 provenance 字段校验过严（见 3.2.4 第 18 条）。
- highres fallback 结果的 `current_inference_small_validation_summary=null` 纯报告遗漏。
- selective promotion 的 `native_missing` 分支。

这些修复只证明代码能够按合同运行，不能把 smoke/静态测试当成性能提升。

<a id="sec-6-4"></a>

### 6.4 适用边界与禁止表述

**可以表述**

- 冻结 DINOv2 语义分支恢复了暗光大目标的语义发现与分类排序；完整顺序推理中 scope 外严格保留 BrightAug，因此非目标域能力被隔离。
- source-selected 因果框稳定器恢复了连续输出的几何稳定性，且不改变中心、分数、排序和输出存在性。
- 高分辨率 S7 补足了 `seq03_small` 的候选覆盖（R@100 `55/64 → 64/64`）。
- 联合前端 `crane_symeood_dino_unified_v1` 已实现 SymEOOD 主头与冻结 DINO 候选的联合推理结构。

**不能表述**

- 不能声称已解决完整 real test 的所有 MCML，或已解决远距离/小目标候选生成。
- 不能声称 frozen DINO 分支已完成不依赖 target-derived scope 的最终部署方法。
- 不能把 `seq03_small` 的候选覆盖改善说成最终排序已解决。
- 不能把 teacher-force / oracle / 手工时序权重结果称为可部署改进。
- 不能把不同模型、checkpoint 或证据层级的结果拼接成同一个「最终模型」结论。
- 不能把三段 fixed target-dev 拼成一个模型的总成绩；不能只报告平均 RIoU。
- 不能把 target-dev 或完整 test 数字写成未知序列泛化或部署结论。
- 不能把「模型改进基本停止」理解为某个候选已通过全部验收；本支线正式部署模型仍是 native S14 `alpha=0.5`、S7 disabled。
- **不能把某个配置的未验证写成整个项目未完成**：Base V3 观测主线已实现图像→OBB 真实在线推理入口与固定 TEST 指标链（见 OBB 文档第 4 节）；「统一入口未完成」「完整 final test 未完成」只在 DINO/S7 支线内成立。
- **不能把图像平面 OBB 输出写成已验证的三维物理状态**：本支线只输出旋转框，不输出物理摆角或米制三维位置。

**表述必须带支线与日期**

本文档中「正式部署模型」「当前最值得继续」「下一步只允许」等判断都带有适用范围与记录日期，引用时必须一并保留：

| 表述 | 适用支线 | 记录日期 / 依据 |
| --- | --- | --- |
| 正式部署模型 = native S14 `alpha=0.5`、S7 disabled | 冻结 DINO/S7 检测支线 | 2026-08-07（来源 D 第 10.17.5 节） |
| Base V3 epoch9 为 OBB 观测主线前端 | OBB 观测支线 | 2026-09-10（来源 R 第 1 节） |
| V4+seq11-v2 `epoch_10` 为本支线内当时最值得继续 | 检测融合 seq11 支线 | 2026-09-05（来源 F 第 14 节） |
| 联合前端仅实现、未验证 | `crane_symeood_dino_unified_v1` | 2026-08-08 记录 + 本轮配置核查 |

**系统边界**

纯 RGB 单目视觉，不使用 PLC、编码器或其他现场传感器；不把图像方向等同于机械自旋角或物理摆角。SymEOOD PQA/quality-primary、QFL、平台 crop 分类器、平台 K sweep、手工时序权重 sweep、Retinex 增强、TTA 与 VLM 依赖方法均已封存，不再作为候选路线。

---

<a id="appendix-sources"></a>

## 附录 A 来源索引

原文件已逐字节保存在 [整合前归档](archive/20260915_pre_consolidation/)，只用于追溯，不继续维护。本表是归档与正文的映射；正文已包含理解方法、关键实验和当前结论所必需的全部信息，归档仅承担追溯与完整原文的作用。

| 锚点 | 归档文件 | 原文件开头（历史快照） | 正文去向 |
| --- | --- | --- | --- |
| <a id="source-l"></a>`source-l` | `archive/20260915_pre_consolidation/innovation4_frozen_dinov2_lowlight_large_target.md` | *创新点 4：冻结 DINOv2 语义分支的暗光大目标排序恢复*，更新 2026-07-26 | 3.1 全部；2.2、2.3、5.3(a)(c) |
| <a id="source-d"></a>`source-d` | `archive/20260915_pre_consolidation/innovation4_dino_small_target_progress.md` | *创新点 4 后续：统一 DINO 远距离与小目标改进实验总账*，更新 2026-08-07 | 3.2 全部；4 全部；5.2、5.3(b)、5.5 |
| <a id="source-f"></a>`source-f` | `archive/20260915_pre_consolidation/20260905_symeood_dino_fusion_seq11_full_handoff.md` | *SymEOOD–DINO 检测融合与 seq11 补充数据完整交接*，更新 2026-09-05 | 6.2 全部；6.3；5.1、5.4、5.5 |
| <a id="source-t"></a>`source-t` | `archive/20260915_pre_consolidation/20260802_unified_temporal_candidate_literature_2025_2026.md` | *统一跨尺度因果时序候选选择：2025–2026 文献映射*，2026-08-02 | 4.2、4.4 |

原文件在仓库中的旧路径为 `docs/<原文件名>`（例如 `docs/innovation4_dino_small_target_progress.md`、`docs/20260905_symeood_dino_fusion_seq11_full_handoff.md`、`docs/20260802_unified_temporal_candidate_literature_2025_2026.md`）。旧引用（含记忆文件与其它文档）按归档文件名在本表定位即可；本节保留的 `source-*` 锚点即原来的跨文档引用目标。

整合清单与哈希：[manifest.json](archive/20260915_pre_consolidation/manifest.json)。该 manifest 是 **2026-09-15 首次机械整合的快照**：其中的 `anchor`、`integrated_body_sha256` 字段描述的是当时的章节编号与结构，本文件的语义重写已改变编号与组织方式，因此这些字段不再对应当前正文，只作为归档原文的身份与完整性凭据。本轮（第二轮）的「原锚点 → 新正文位置」映射、状态裁定与待核验清单见 [融合核对表](融合核对表.md)。本次为语义级重写：同一研究背景、结构、符号、数据协议与评价边界只保留一处完整说明；同一实验的多次记录已合并为单一记录；重复的交接说明、自检清单、状态总结与「下一步」链已删除；已被后续裁决取代的计划退出行动清单。原文件中每条独有的方法、公式、实验、裁决和证据入口均在正文有对应位置。本次整理不代表重新验证过权重、外部结果、文献或实验指标。

相关文档：[OBB 观测可靠性与连续输出](OBB观测可靠性与连续输出.md)、[Webots 单目深度估计](Webots单目深度估计.md)、[研究文档入口](README.md)。
