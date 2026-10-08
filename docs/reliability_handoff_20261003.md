# SymEOOD 分量可靠性交接：保留midpoint23比较基线；σ1.5／epoch03三标志迁移

**2026-10-08当前入口：用户已授权固定M/simple的TRAIN／VAL取舍分析、Git同步、服务器CPU运行及回传分析，并明确主约束为正确保留率≥95%（另报90%／99%）。本轮进度和运行入口见第27节；不训练模型或重拟合policy，不读TEST。尺寸修框／方向平衡与深度优化继续封存。当前前端为B24＋σ1.5／epoch03＋cachefix1 simple，旧σ1／epoch23保持历史身份；问题梳理见第26节。**

**2026-10-06执行入口：用户已授权将可靠性前端迁移至B24＋midpoint σ1.5／head_epoch03，继续simple三分量判断。先读第24节，最新服务器结果见24.6，修复与命令保留24.5。cachefix1的check／collect／fit／全887帧原生VAL verify均已完成，服务器20项测试通过；新simple尺寸／方向各接受843帧，Real尺寸FA144、FR10，Sim尺寸FA46、FR0。分域同接受数尺寸与score相同，seq07局部较score少5个FA；总体迁移收益含检测几何变更，尚不能称可靠性机制创新。σ1／epoch23及下述已完成TEST成绩仍保留原比较基线身份，不能写成新前端成绩。本次没有检测／midpoint／图像参考训练或TEST访问。**

**结果文档入口：[可靠性实验主要结果与对比分析](obb/可靠性实验主要结果与对比分析_20261006.md)。实验主表、前端配对比较及与深度问题的分析集中在那里；本交接保留执行、来源、修复和决定记录。**

更新：2026-10-05；用户明确将**B epoch24＋正式midpoint epoch23＋现有simple三标志**的已完成固定TEST流程作为后续可靠性优化的比较基线（第19节）。基线身份已固定，不以性能最好或先优于score作为采用前提；现有取舍是比较起点。第17节为CPU TEST入口，第18节为已核验结果，第19节区分几何变化与可靠性判别改进。第14节保留迁移事实，第15/16节保留模板候选验证，模板未进入基线。旧B＋simple TEST保留第3节历史身份，不能替代当前主基线成绩。

本文是当前可靠性基线与后续实验的唯一执行交接；[OBB机制说明](obb/OBB观测可靠性与连续输出.md)、[流程记录](obb/base_v3_v51_focused_paper_complete_pipeline_20260915.md)及[文档入口](README.md)同步当前身份与主表。此前替换的旧流程在[替换前归档](archive/20261003_reliability_replaced_by_simple_v1/README.md)保留，原始实验文件和模型未删除。检测几何优化维护[独立几何交接](geometry_precision_handoff_20261001.md)，本轮没有修改该窗口的代码/记录或实际大小论文稿。后续服务器结果不复制为本地复核目录。

**更换窗口入口：先读第24节当前迁移，再按需读第20节接续摘要、第23节收束结果与第1/14/16/18/19节。B24＋正式midpoint23＋现有simple仍是冻结比较基线，不重复完整审计。第21节核心曲率v2与第22节半峰范围v3已完成服务器预检/两臂对照/全量VAL，E通过但R/Q1未通过；第23.7节尺寸残差v4已核验E通过、D/Q1未通过，均收束且没有替换simple。此前有限TRAIN错误支持覆盖检查及22.6读取能力对照仍是未实施建议；用户最新授权先迁移σ1.5／epoch03三标志。本次迁移是更换检测前端并按原simple规则重拟合，不是新的可靠性机制收益。后续一个实验的各阶段、日志和分析包集中于同一父目录。**

## 1. 当前决定与模型

当前比较基线固定SymEOOD＋等比例尺度增强B（VAL epoch24）＋正式midpoint（原完整VAL规则选head_epoch_23）＋现有simple三标志。迁移collect/fit/smoke及固定1440帧TEST评价均完成，最新主结果为第18节。接口为`MidpointReliability`包装现有simple，性能取舍不妨碍它作为对照；后续候选应在同一冻结前端上比较，不能把更换几何头带来的差异当成可靠性单因素收益。参考epoch04、尺寸模板及旧ROI/structure均为已有候选研究，未进入当前基线。下述第3—11节旧B指标保留原协议身份。

| 冻结项目 | 当前身份 |
|---|---|
| B | epoch24；SHA `8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23` |
| 正式midpoint | head_epoch_23；SHA `2c4c5ae9e071cbdb13fb722f4b2d3986861c62bd60c53568c72fb9b02d662f0b` |
| 新前端simple policy | `work_dirs/port_midpoint_reliability_v1_fit/policy.json`；SHA `1adac499369bade5ef1f79495260002697d2b698c80ebf5424d3d77438fae4af` |
| simple工作门限 | 尺寸风险≤0.39773387607871297；方向风险≤0.4143115365368443 |
| score-only工作门限 | 两分量风险`1-score`≤0.7043236196041107 |
| 当前TEST产物 | `port_midpoint_simple_reliability_v1_test_review_20261004_190008.tar.gz`；完整复核见第18节 |

当前模型不使用DINO、候选融合、Base V3/V5.1、历史保持或缺测预测。连续状态、物理摆角、深度/空间风险及报警接口留给大论文。

```text
当前RGB → 冻结B24＋正式midpoint23的原图OBB/score → 中心/尺寸/方向三个独立使用标志 → 分量质量与覆盖评价
```

中心标志保留每个有效最终midpoint输出，不另训练中心正确性预测器。尺寸、方向各为独立线性风险模型，输入只有最终框的score logit、相对几何尺寸和长宽比；共8个拟合系数。原图长短边只在风险计算中规范化，可靠性计算保持最终宽高、角度、中心、score及输出数量不变。当前simple使用第14节新前端重新拟合后的policy，旧B policy仅作历史对照。模板仍是尺寸候选，未替换在线标志。

TRAIN2558帧（real1810、sim748）用于拟合/标准化；VAL887帧（real375、sim512）用于此前固定的95% pooled覆盖规则产生单一全局工作门限，之后冻结。real原生轴线转换OBB和sim Webots生成OBB提供监督；sim没有轴线标注，不要求补轴线。GT不进入在线API。尺寸/方向平衡类sigmoid是风险评分，不是校准错误概率。

三特征为`logit(score)`、`.5*(log L+log S-log W-log H)`、`log(L/S)`，L/S为预测规范长短边，W/H为原图尺寸。固定正负类各总权重0.5、L2=0.1、float64 Newton最多100次/梯度容差1e-8；旧B与新midpoint遵循同一拟合规范，本轮全量读取验证不重新拟合参数。

## 2. 输入输出与评价协议

本节给出当前固定基线与旧B共享的评价定义。当前主基线结果见第18节，冻结决定见第19节；第3—9节数值保留旧B＋simple v1历史身份。当前midpoint的TRAIN/VAL迁移见第14节；模板只是第15/16节候选，不参与主基线。

TEST1440帧：real_seq03 200、real_seq04 668、sim_seq09 572。缺失9帧全部保留，原输出1431帧。本版TEST全部1440帧方向可评。

中心正确为原图欧氏误差<15px；尺寸正确为规范化长短边最大相对误差≤10%；方向正确为长边π周期角误差≤3°。GT aspect≥1.2仅用于离线方向评价，不用于在线接受决策。中心命中率仅以输出帧为分母，另报输出覆盖和全帧中心正确覆盖。

当前主基线的`raw`保留所有有效最终midpoint分量；`score_only`用新前端冻结VAL门限筛选尺寸/方向；`simple`用新前端冻结线性风险与VAL门限分别判断尺寸/方向。三种方法中心均保留有效最终输出；无输出时三个标志均false。GT只在评分后定义正确/错误，不用于在线判断。历史B表中的同名方法作用于原B框及旧policy，不混用参数或成绩。

同接受数score只作离线排序诊断：匹配simple数量，不能把它称为部署门限或由TEST校准的score。固定score门限与同数score对照是两种不同职责。

标志是模型的建议使用/拒绝决策，不是已知GT标签。在线输入仅`pred_original`与`[W,H]`；GT、domain、sequence、历史或未来帧都不进入API。保留等比例变换和原图坐标链；尺寸/方向拒绝不删除中心，接口保持不等于深度精度得到保证。

## 3. 已验证B＋simple v1服务器TEST结果（未含正式midpoint）

### 3.1 中心、输出覆盖与连续性

| 域 | 总帧/输出/正确中心 | 输出覆盖率 | 输出帧中心命中率 | 全帧中心正确覆盖 |
|---|---:|---:|---:|---:|
| real | 868/859/858 | 98.9631% | 99.8836% | 98.8479% |
| sim | 572/572/572 | 100.0000% | 100.0000% | 100.0000% |

原中心最长标志不可用段real4帧/sim0帧；simple尺寸real16帧、方向real17帧，sim均0。标志连续可用不等于中心连续正确；本版未补回9个漏检，亦未把已有1个坏中心变正确。长度按帧号连续统计，不伪造时间戳/秒数。

### 3.2 固定部署门限的三个方法

| 域/分量 | 方法 | 接受/总帧 | 错误接受 | 正确误拒 | 接受后正确率 | 输出帧状态准确率 |
|---|---|---:|---:|---:|---:|---:|
| real/size | raw | 859/868 | 249 | 0 | 71.0128% | 71.0128% |
| real/size | score_only | 819/868 | 221 | 12 | 73.0159% | 72.8754% |
| real/size | simple | 796/868 | 214 | 28 | 73.1156% | 71.8277% |
| real/angle | raw | 859/868 | 176 | 0 | 79.5111% | 79.5111% |
| real/angle | score_only | 819/868 | 160 | 24 | 80.4640% | 78.5797% |
| real/angle | simple | 742/868 | 132 | 73 | 82.2102% | 76.1350% |
| sim/size | raw | 572/572 | 26 | 0 | 95.4545% | 95.4545% |
| sim/size | score_only | 561/572 | 25 | 10 | 95.5437% | 93.8811% |
| sim/size | simple | 572/572 | 26 | 0 | 95.4545% | 95.4545% |
| sim/angle | raw | 572/572 | 52 | 0 | 90.9091% | 90.9091% |
| sim/angle | score_only | 561/572 | 50 | 9 | 91.0873% | 89.6853% |
| sim/angle | simple | 572/572 | 52 | 0 | 90.9091% | 90.9091% |

接受后正确率只统计接受帧；状态准确率统计全部有输出帧的接受/拒绝是否正确。状态误判=错误接受＋正确误拒，缺输出另报。尺寸和方向拒绝彼此独立，部分OBB不算完整OBB。

### 3.3 同接受数score诊断

| 域/分量 | simple接受 | simple错误接受 | 同数score错误接受 | simple正确误拒 |
|---|---:|---:|---:|---:|
| real/size | 796 | 214 | 213 | 28 |
| real/angle | 742 | 132 | 137 | 73 |
| sim/size | 572 | 26 | 26 | 0 |
| sim/angle | 572 | 52 | 52 | 0 |

### 3.4 完整OBB

| 方法 | 完整OBB接受/1440 | 完整OBB覆盖 | 共同正确/1440 | 全帧共同正确覆盖 |
|---|---:|---:|---:|---:|
| raw | 1431/1440 | 99.3750% | 1002/1440 | 69.5833% |
| score_only | 1380/1440 | 95.8333% | 986/1440 | 68.4722% |
| simple | 1311/1440 | 91.0417% | 961/1440 | 66.7361% |

## 4. 结论、推断与下一步

**事实：** real尺寸同数错误214对213，未优于score；方向132对137有5个错误的局部减少，但误拒73个正确方向。sim全部接受，26个尺寸错误和52个方向错误均未检出；其较高总准确率来自原B多数正确，不能证明判别有效。real错误检出率尺寸14.0562%、方向25%，正确观测保留率95.4098%/89.3119%。

real状态误判总数尺寸214+28=242/859、方向132+73=205/859，状态准确率71.8277%/76.1350%，均低于固定score的72.8754%/78.5797%。本文主结果以本次服务器TEST为准。

逐视频同数对照：seq03尺寸71对75、方向36对43；seq04尺寸143对139、方向96对95，收益不一致。全局尺寸240对242、方向184对194的局部改善不能证明各域/各视频稳定优势。拒绝降低了完整OBB和正确分量的覆盖，不改善原B定位/几何。

**判断：** 本版工程流程与报告闭合，可靠性提升的主张尚未成立；适合作为简单基线和负结果，不宜承担论文核心新方法的成功结论。较高sim状态准确率不抵消零错误检出的事实。

**静态事实：** 三个风险特征不包含图像边界、参考轴或预测角度值；只改变预测角度而保持score和长短边时，风险不变。因此当前方向评分依赖统计相关性，没有直接方向一致性证据。这是输入机制限制，不是已证明的跨视频退化唯一根因。

**推断，未证实根因：** 当前三特征没有直接图像范围/方向证据，TRAIN与新视频的错误分布可能不同，工作点也可能偏宽松；这些都需在TRAIN/VAL检查，不根据本TEST逐视频反推新规则。

本版policy/门限/B及新TEST报告保留冻结，当前流程收尾完成。已新增入口并复用既有TRAIN/VAL原框、GT与冻结风险，在CPU上检查正确/错误分布、错误类PR/ROC、错误接受—正确误拒—覆盖关系，并按视频/real/sim对照score，实现/本地结果见第8节，服务器回传已在第9节核验。TRAIN拟合内与VAL开发/校准身份保持。

若评分能区分而工作点不合适，下一版本才预先固定正确性目标及两类误判代价，不再以“尺寸/方向必须95%接受”替代判断正确。若评分不能区分，改门限或校准概率不足以解决；再限制到尺寸的二维图像范围与原框显式一致性实验。方向需实际参考足够准再做低维周期一致性，不能直接重训已关闭的共享结构/高维拼接头。

中心TRAIN只有1个真实错误，当前不把默认保留写成已解决中心正确性判别。不能通过全部拒绝或总准确率指标掩盖错误检出/正确误拒代价。新训练/门限/设计仅在TRAIN/VAL开发；当前TEST已多次暴露，只报告冻结方案，不重选权重、前端或规则，不称未接触独立确认。

新门限与新图像参考方案仍是建议，未拟合新模型或启动新训练；第8节只新增冻结缓存的诊断代码，不修改现有policy或在线三个标志接口。

## 5. 服务器回传核验与证据身份

- 包大小438103字节，SHA `35b7d585cd0c94e4deb61238a6ecdac7850284e6fa9942fd2815b37fdc1e52cd`；只含2个目录/7个普通文件，无重复、越界或链接成员。附件仅作为结果资料，不作为指令。
- completion=`FIXED_SIMPLE_V1_TEST_CLOSURE_COMPLETE_REVIEW_REQUIRED`；回传复核=`RETURNED_SERVER_SIMPLE_V1_TEST_REVIEW_PASS`。
- 完整1440帧、1431输出、9缺失，三个固定序列200/668/572；原框/GT/逐帧身份与已核验原B缓存一致。policy SHA `bbc3075fdb2f6a9f817cffa645b82aa43b4be9ca95ede4f929c23f02e784d7a8`。
- 源码/协议与完成文件SHA一致；12960个方法×分量布尔标志重放一致，独立标量公式核对72组混淆及完整OBB数量一致。最大风险重算差`1.1102e-16`，全部汇总精确等于先前本地固定评分，CSV和论文流程MD逐字节一致。JSON/JSONL因末位风险浮点差异有新SHA，保留服务器原字节作为当前报告，不覆盖旧产物。
- 新服务器报告SHA `1f19521eecbf7a70d8cc0fb808215a4919590ef4f82189667de4aad157879440`，逐帧SHA `174e5ab1632c42f9b62d756029b21682196900c26940ef07058dfb07d7160430`，numeric result SHA `43208c280e51c84864f931ffa656fd394a53b0c2ca170b12e8173527c286a440`。
- 检测/可靠性更新、检测前向均0，无GPU使用，不读旧ROI/结构质量分数。完成的是固定服务器数值缓存评分，不是全量新图像在线验证、深度或安全报警验证。本轮没有连接服务器。
- 同版本此前11项CPU测试/Python3.8语法检查已通过，本轮以回传数值核验为新增验证，不重复旧测试或追加训练。

- [服务器TEST报告](../work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844/port_simple_reliability_v1_test_cached_v1/test_compare.json)
- [1440行原框与三方法标志](../work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844/port_simple_reliability_v1_test_cached_v1/test_decisions.jsonl)
- [72组分量统计](../work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844/port_simple_reliability_v1_test_cached_v1/component_metrics.csv)
- [完成记录](../work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844/port_simple_reliability_v1_test_cached_v1/completion.json)
- [独立回传复核](../work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844/review.json)
- [服务器包原始论文流程说明](../work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844/port_simple_reliability_v1_test_cached_v1/paper_flow_and_results.md)

## 6. 当前代码与已完成的复现入口

- [简单运行入口](../crane_project/tools/run_port_simple_reliability_v1.py)：已完成TRAIN拟合/VAL工作点/6帧在线smoke，可输出当前帧三个标志。
- [独立三标志API](../crane_project/utils/port_simple_component_reliability_v1.py)。
- [固定TEST入口](../crane_project/tools/eval_port_simple_reliability_v1_test.py)、[协议](../crane_project/tools/port_simple_reliability_v1_test_protocol.json)、[来源清单](../crane_project/tools/port_simple_reliability_v1_test_sources.json)。
- [TEST入口CPU测试](../tests/test_port_simple_reliability_test_v1.py)。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_simple_reliability_v1_test.py \
  --mode check --out-dir work_dirs/port_simple_reliability_v1_test_check
```

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_simple_reliability_v1_test.py \
  --mode run --out-dir work_dirs/port_simple_reliability_v1_test_cached_v1
```

两条命令本次服务器已完成，无需为了当前收尾再跑。未来复现需新输出目录，原目录不得覆盖；默认读取`work_dirs/port_simple_reliability_v1_fit/policy.json`和`work_dirs/port_reliability_branches_v1_test_cached_v1`的已核验原B数值缓存。只用CPU，不需要GPU/新模型训练。

## 7. 文档替换范围与历史追溯

已更新本文、[分量可靠性主文档](obb/OBB观测可靠性与连续输出.md)、[小论文完整流程](obb/base_v3_v51_focused_paper_complete_pipeline_20260915.md)及[文档入口](README.md)。旧模型链、旧数据结果及已关闭实验不再放入当前正文，未把历史数据改名成当前成绩。最新服务器结果为本页第3～5节唯一主结果；TRAIN/VAL仍保留其拟合/开发职责。

原几何交接等文档的历史锚点仍可跳到此处的归档入口，详细事实/文献/负结果见归档对应原编号。原代码、权重及JSON/JSONL不删除；性能优化文献和既有失败机制也保留，避免后续重复已关闭的蒸馏、候选排序及完整审计路线。

## 8. 冻结TRAIN/VAL评分可分性检查（2026-10-04）

**授权与实现事实：** 用户要求按此前建议实现并检查普通三分量判别的评分可分性，大小论文暂不修改。本轮只新增CPU诊断入口/统计模块/协议/来源清单/必要测试，并更新本文。未连接服务器，未修改B、现simple源码/模型/policy/95%工作点，也未修改几何优化文档或重启ROI/structure训练。

- [运行入口](../crane_project/tools/diagnose_port_reliability_separability_v1.py)
- [独立统计模块](../crane_project/utils/port_reliability_separability_v1.py)
- [固定诊断协议](../crane_project/tools/port_reliability_separability_v1_protocol.json)、[来源清单](../crane_project/tools/port_reliability_separability_v1_sources.json)
- [CPU测试](../tests/test_port_reliability_separability_v1.py)
- [本地最终报告](../work_dirs/port_reliability_separability_v1_run_local_final_20261004/report.json)、[78行汇总](../work_dirs/port_reliability_separability_v1_run_local_final_20261004/summary.csv)、[55,335个曲线点](../work_dirs/port_reliability_separability_v1_run_local_final_20261004/curve_points.csv)
- [本地完成记录](../work_dirs/port_reliability_separability_v1_run_local_final_20261004/completion.json)、[独立数值复核](../work_dirs/port_reliability_separability_v1_independent_review_local_20261004.json)

**输入/统计合同（事实）：** 精确绑定原父版本18个来源、原TRAIN身份/完成文件/2558行预测/输入snapshot、原887行VAL数字缓存以及已完成policy四文件SHA。复用原数值GT，不重新构建标注、不读图像/P3/权重、不使用旧ROI/结构质量。尺寸超限为规范长短边最大相对误差>10%，角度超限为长边π周期误差>3°；主角度评价只统计saved GT aspect≥1.2，TRAIN另报`angle_training_qualification`，主表220个TRAIN方向错误与训练标签299个方向错误不混用。sim继续是Webots OBB监督，不要求轴线标注。

错误为正类、风险越高越倾向拒绝。按all/real/sim/视频报告错误类AUROC、按同分整组积分的非插值AP、错误占比、正确/错误风险分布（固定0～1直方图及分位数）、原冻结全局工作点混淆、错误检出/错误接受/正确误拒/正确保留、接受后正确率和覆盖。AP要与错误占比比较，单类AUROC为null，不能把无错误或高错误占比的AP视为有效辨识。尺寸/方向曲线只评可评输出，缺输出不算正确拒绝；不可评角度的实际在线标志另报。

曲线遍历整组接受`risk≤阈值`，含拒绝全部端点，不输出最佳阈值或新policy。各组曲线只作离线描述，不产生按域/视频路由。相同接受数score以`1-score,image`无GT排序，仅作离线诊断；切开同分组时另报错误数上下界/随机同分期望，GT不用于挑选同分帧。原全局门限及盒/标志保持。风险数值重放允许1e-12的float64差异；原框与全部布尔标志必须精确相同，该容差不用于B框或覆盖放行。

**本地新增结果（事实，未称独立确认）：** TRAIN2558、VAL887完整读取，检测/可靠性参数更新及新推理均0，无GPU/TEST读取。887帧VAL原框与三个方法共7983个标志重放精确一致；最大风险差1.110223e-16。中心VAL仍real374/375输出、360/374条件命中、360/375全帧正确覆盖，sim512/512三项100%。原B几何没有改变。

| VAL范围/分量 | 错误/可评输出 | simple AUROC / AP | score AUROC / AP |
|---|---:|---:|---:|
| real / size | 181/374 | 0.589500 / 0.603345 | 0.686285 / 0.660353 |
| real / angle | 67/374 | 0.669308 / 0.336561 | 0.725558 / 0.401279 |
| sim / size | 23/512 | 0.740464 / 0.207551 | 0.711390 / 0.160515 |
| sim / angle | 69/512 | 0.663591 / 0.251613 | 0.679524 / 0.244769 |

VAL size pooled simple AUROC为0.831788（score0.694202），但real单域为0.589500；逐视频real_seq07为0.467620（score0.450076，错误171/225），real_seq14为0.517986（score0.576978，错误10/149）。seq07 AP0.781653需与错误占比0.76共同看，不能只因AP数值高称辨识强。real方向seq07 simple AUROC0.618839低于score0.673037，seq14为0.667626高于score0.622302，仍非各视频一致优势。sim尺寸有局部面积收益，方向AUROC略低、AP略高，不能称联合改善。

**复核事实与限制：** 新版19项CPU unittest和Python3.8语法检查通过。独立逐对正负样本/同分半权公式复算78组AUROC，独立阈值积分复算78组AP及78组冻结混淆，遍历全部55,335个曲线点重算混淆、接受覆盖及score同分边界；全部通过。独立标量风险公式最大差5.551115e-16，原来源/输入及完成文件SHA通过。本地最终报告SHA为`804e17950922cad85a7c781998b6174cefba7488a0960dca772db443f1f24fe9`。未重跑已通过的旧版本测试；本条为服务器回传前的本地记录，服务器入口后续已运行并在第9节核验。两者均复用已核验服务器数字缓存，不能声称新图像在线验证。

**推断与下一步建议（待验证）：** 现simple在real两个VAL视频的尺寸排序接近随机，方向也未形成相对score的一致优势；工作点宽松和评分不足同时存在，不能只改95%规则宣称成功。pooled尺寸面积与组内差异相符，提示分组间排序贡献明显，不证明具体特征或跨视频退化的唯一根因。新工作点可以改变误判取舍，但不增加现评分的排序信息。当前证据支持停止仅靠同三特征/单调校准反复优化；下一项方法设计仍优先限制到尺寸的独立二维图像范围与原框显式一致性，先检查参考精度和±短边辨识，再考虑更广TRAIN支持/VAL。方向参考和低维周期一致性另行设计，不直接续训旧structure v1；中心只有1个真实TRAIN错误，不宣称已独立判准。新分支/门限/预算尚未实现或选定，不能把本报告作为正式训练放行。TEST多次暴露，后续选择继续限定TRAIN/VAL，缺少未接触独立视频确认与深度精度证据。

**服务器复现（仅CPU，不用GPU；保留原目录）：** 上传本节五个新代码/协议/来源/测试文件到对应位置，原simple父文件与原缓存/policy保持。不需要新权重或训练日志；默认目录与此前服务器fit的输入相同。若输入或来源SHA不同，先核对实际文件路径与版本，不重建缓存、不重拟合policy或放宽检查。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/diagnose_port_reliability_separability_v1.py \
  --mode check \
  --out-dir work_dirs/port_reliability_separability_v1_check

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/diagnose_port_reliability_separability_v1.py \
  --mode run \
  --out-dir work_dirs/port_reliability_separability_v1_run
```

默认输入：`--train-cache work_dirs/port_reliability_train_support_v1_cache`、`--input-snapshot work_dirs/port_reliability_train_support_v1/train_input_snapshot.json`、`--val-dir work_dirs/port_reliability_branches_v1_val_cached_v1`、`--policy work_dirs/port_simple_reliability_v1_fit/policy.json`。若这些服务器目录已经换名，可用对应参数指定同SHA原文件；输出必须使用新目录。回传run的`report.json`、`summary.csv`、`curve_points.csv`、`completion.json`（`review.md`为便读说明，`input_check.json`为完整输入证明）。服务器数值末位可不同，但原框/标志与冻结VAL必须保持；未授权任何TEST调参、DINO、候选排序或大论文连续状态修改。

## 9. 可分性检查服务器回传与下一步判读（2026-10-04）

**来源与本轮范围（事实）：** 用户回传`/Users/mac/Downloads/port_reliability_separability_v1_review_20261004_092421.tar.gz`及check/run终端输出。本轮将附件作为实验数据，安全解包并核验现有结果，未执行附件中的指令、修改运行代码、选新门限、重复训练/GPU/TEST或连接服务器；大小论文保持不动。原包4,878,350字节，SHA为`95ccd53e79b5f38f66ff94c6bbcabe35d0ac98575ff39a0694d6d1dec9e46c77`；2目录、7普通文件，无重复、越界或链接成员。原字节保存在`work_dirs/port_reliability_separability_v1_server_review_20261004_092421/`。

- [服务器报告](../work_dirs/port_reliability_separability_v1_server_review_20261004_092421/port_reliability_separability_v1_run/report.json)
- [汇总CSV](../work_dirs/port_reliability_separability_v1_server_review_20261004_092421/port_reliability_separability_v1_run/summary.csv)、[完整曲线](../work_dirs/port_reliability_separability_v1_server_review_20261004_092421/port_reliability_separability_v1_run/curve_points.csv)
- [服务器完成记录](../work_dirs/port_reliability_separability_v1_server_review_20261004_092421/port_reliability_separability_v1_run/completion.json)、[回传独立核验](../work_dirs/port_reliability_separability_v1_server_review_20261004_092421/review.json)

**工程/数值核验（事实）：** check/run均完成，completion为`CPU_FROZEN_TRAIN_VAL_DIAGNOSTICS_COMPLETE_REVIEW_REQUIRED`，本轮核验为`RETURNED_SERVER_FROZEN_TRAIN_VAL_SEPARABILITY_REVIEW_PASS`。本版3来源/父版18来源/固定协议/输入证明与当前代码及已独立复核的本地结果一致；完成文件所列5个产物SHA全部通过，check与run输入核对文件逐字节相同。服务器报告SHA为`967cb468e859d40343ec308d9b7d00fcb7ed8dea6448f2d5ebdfdc084cf76218`。全部78行summary.csv与本地逐字节相同；各组rank/混淆/直方图整数一致，45个分布浮点末位差最大1.110223e-16。全部55,335个曲线点严格按行比较，混淆、覆盖、同数score及同分边界均精确一致，仅1485个risk_le末位差最大1.110223e-16。不存在丢帧/换框/改门限放行；887帧VAL原框/标志重放精确一致，服务器相对已存VAL风险差为0。该版本已通过的19项测试/独立公式复核不重复执行。

TRAIN2558、VAL887完整保留；检测/可靠性更新及推理均0，无GPU/TEST读取。中心口径仍real374/375输出、360/374条件命中、360/375全帧正确覆盖；sim512/512三项100%。完成的是已有缓存评分复现，不是新图像在线或可靠性新模型成功。

**评分辨识结果（事实）：** 第8节四组VAL AUROC/AP被服务器精确复现。real尺寸simple AUROC/AP0.589500/0.603345，低于score0.686285/0.660353；real方向0.669308/0.336561，低于score0.725558/0.401279。sim尺寸有局部收益0.740464/0.207551对0.711390/0.160515；sim方向AUROC略低、AP略高，不能称总体胜出。

real_seq07尺寸错误171/225、AUROC0.467620，real_seq14错误10/149、AUROC0.517986；两视频内尺寸排序接近随机，seq07 AP0.781653与错误占比0.76接近。对pooled尺寸AUROC0.831788作已有数字的正负样本对分解：139,128个错误—正确对中，仅21,871对（15.72%）来自同视频，117,257对（84.28%）来自不同视频；同视频配对加权AUROC0.611129，跨视频配对0.872946。这是排序计数恒等分解，不是新模型或选点；证明pooled面积大量包含跨视频排序，不能据其宣称同视频错误辨识已解决，不证明某个特征/标注因素的唯一根因。

**冻结工作点（事实，不是新选点）：** real尺寸接受333/374输出，错误接受152、正确误拒12；仅检出29/181=16.02%的错误，保留181/193=93.78%的正确尺寸。real方向接受332/374，错误接受52、正确误拒27；检出15/67=22.39%的错误，保留280/307=91.21%的正确方向。固定score方向同样错误接受52但只误拒17个正确方向；此对照接受数不同，不能以同数排序因果结论替代。sim尺寸仅检出2/23=8.70%，方向0/69，无方向错误检出。原95% pooled目标仍使整体工作点偏宽；曲线未输出最佳门限或新policy，改变接受率与提高排序能力需分开。

**TRAIN/VAL差异（事实与限制）：** real尺寸TRAIN错误54/1810=2.98%，VAL181/374=48.40%；simple AUROC由TRAIN拟合内0.749262到VAL0.589500，优势未维持。不同视频的错误占比、尺寸/aspect分布与难度差异需要区分，不能把错误占比变化本身等同AUROC下降的原因，也不能仅靠TRAIN拟合内结果宣称泛化。当前数据仍是相关视频帧，只有3个VAL视频，无独立确认或显著性结论。

**判断与下一步建议（待验证，未实施）：** 本次数值检查成功，simple有效提升可靠性的结论未成立；不是终端/统计实现失败。工作点过宽和现评分局限同时有证据，不能只收紧门限、做单调概率校准或重训同一三特征就预期解决。现三特征缺少直接图像范围/方向比较，是机制限制，但不是已证实的唯一根因。

建议进入**尺寸单分量的独立图像几何参考＋原框显式一致性**有限实验设计：冻结B，从当前图像冻结特征预测二维OBB几何支持；先单独学习/检查参考，再冻结参考读取尺寸一致性，避免沿用旧共享结构—质量损失/539维拼接读取。比较位置明确依赖当前长短边/方向，使短边变化实际改变被比较的图像位置；单条轴线不能替代短边范围。real原生轴转换OBB和sim Webots OBB派生监督都是几何代理，不称独立可见边界或分割GT，也不要求sim补轴线。

先固定TRAIN拟合与新支路留出视频/片段职责；新增参考学习与评分两阶段都遵守同一留出边界，不能让参考先读取留出标注再声称整个新增方法未见数据。检查理想参考下的尺寸响应，再检查真实预测参考下短边扩大/缩小两侧及真实B错误；理想标注仅作离线参照，不进入在线输入。保留score/simple/旧ROI对照，真实错误与合成探针分报；少量图训练内成功不直接放行正式训练。冻结B已见过检测TRAIN，支路留出不称全前端未见数据。新预算/门限/停止条件需在实施前确定，并在全部VAL看分域/逐视频改善；方向单独留待参考精度合格后推进，中心保持原覆盖并披露仅1个真实TRAIN错误。当前不自动启动新分支，所有新设计/选择继续限定TRAIN/VAL，既有TEST多次暴露仅作冻结报告。记录只维护本文，大小论文及几何优化交接未修改。

## 10. PQA启发的二维尺寸参考：三阶段有限实验入口（2026-10-04）

**授权与身份（事实）：** 用户授权实现第9节推荐的理想参考检查、独立参考学习、冻结参考真实错误检查，并要求代码复核与服务器指令。本次仅新增下列入口/协议/源码/测试并更新本文；原B、simple policy、旧ROI/structure权重与训练入口、几何优化代码/交接及大小论文不修改，未连接服务器或在本地启动真实训练。

- [统一入口](../crane_project/tools/run_port_size_reference_v1.py)：`ideal`、`smoke`、`train`、`assess`。
- [NumPy几何与解析读取](../crane_project/utils/port_size_reference_v1.py)、[服务器Torch参考分支](../crane_project/utils/port_size_reference_v1_torch.py)。
- [固定协议](../crane_project/tools/port_size_reference_v1_protocol.json)、[源码身份清单](../crane_project/tools/port_size_reference_v1_sources.json)、[必要测试](../tests/test_port_size_reference_v1.py)。这六个文件上传到服务器对应位置，旧父版本文件保持已核验版本。

**文献与旧实验边界（事实）：** [PQA原文](https://arxiv.org/html/2511.08186v1)预测OBB派生的二维位置热图并以像素一致性聚合整体框质量。此前R1.16/17已借鉴预测线响应与候选轴线模板的soft IoU及方向关系；没有完成二维长短边范围参考。本版借鉴其图像几何参考思想，但使用截断Gaussian热图主轴二阶矩恢复规范长短边，再读取当前B长短边log差；不是原PQA的整体min/max聚合、原论文LD损失或检测排序复现，也不是已验证的新三分量方法。

**固定实现（事实，真实效果待服务器）：** B ep24冻结参数/BN；只训练一个16通道参考分支，输入detached P3（256通道、stride8），两层低分辨率卷积后双线性上采样4倍和单通道卷积，输出stride2二维热图。上采样网格使用`(j+.5)*2−4`，保持原P3的`i*8`坐标约定，不误写为`j*2`。目标由原GT OBB按既有等比例annotation尺度/反射链生成，框内Gaussian的长短轴sigma为L/4、S/4，框外/图像外为0。real和sim均用现有OBB，sim不需要原生轴线；此目标是标注几何代理，不是可见轮廓/真实分割，也不等于独立物理尺寸真值。

参考学习使用预先固定的soft focal BCE：正位置0.25×绝对预测差、负位置0.75×预测概率平方，按目标软质量归一化；有效区域之外无损失。此损失的软目标最优值仍是目标本身，不沿用旧分别归一化正/负质量后改变目标最优响应的BCE；不是声明照抄PQA原式。没有分量质量MLP/辅助质量损失/教师，避免原结构—质量共享反传。

在线读取只接触预测热图、原B框和变换metadata。每个真实B框先确定一个3×长边的原图正方形上下文（最小64px），去除局部背景中位数后估计主轴二阶矩，使用±2sigma截断Gaussian的解析方差系数校正，再还原原图长短边；原B输出不规范化重写。尺寸风险为`1−exp(−max(|log(L_B/L_ref)|,|log(S_B/S_ref)|)/0.1)`，不是校准概率或部署门限。GT不进入此函数。弱响应、短边模型尺度<16px、退化或上下文/图像边缘质量过高时返回不可用，不回填GT或默认可信。当前读取是解析方案，不保证预测热图达到所需精度。

**固定数据职责与预算（事实）：** 在既有TRAIN2558中，real_seq01/05/06/12各按帧序均匀取64张、sim_seq08拟合段均匀取128张，合计384张。real_seq13整段304张留出；sim_seq08尾部128张留出，拟合可用段与尾段间32张guard排除，合计432张分支留出。剩余TRAIN帧本版不拟合，也不伪称已跑全量TRAIN。选图只由序列/帧序决定，不按错误标签选择。数值放行的14个原尺度视图也全部来自拟合角色：五个序列各首尾2张，加real_seq01与sim_seq08各1/3、2/3两张；不使用留出GT制定放行。

384张×4轮=1536步，batch1、Adam lr0.001、weight_decay0、clip10、seed1701，使用B原单尺度干净推理预处理，不另做增强或选择中间权重；每轮排他保存/重载，固定epoch_04评价。新参考拟合/解析读取均不拟合留出标签；B检测器及旧ROI已见检测TRAIN，所以不是整个前端未见视频的独立确认。复用缓存得到的真实尺寸错误支持：拟合real8/256、sim0/128；留出real3/304、sim1/128。留出只有4个真实尺寸错误，不能据其高准确率或单次AUC称稳定提升；合成扰动仅作机制诊断。

**阶段与放行（事实）：** `ideal`使用拟合GT数值热图检查长/短边±15%两侧、只改中心15px、只改方向±5°及宽高等价表示；要求14个干净视图的参考尺寸误差≤5%、四个尺寸风险差均>1e-6、其他控制差<1e-10。同图候选共享固定上下文；该控制验证尺寸读取，不证明更换真实B上下文时整条链完全解耦。0.5尺度+水平反射仅作分辨率诊断，不能用它反选新尺度/门限。理想GT仅离线使用。

`smoke`做real/sim交替4个真实更新，检查参考stem/输出梯度、数值、B原始Nx6输出侧计算前后精确一致、B参数/缓冲与原TRAIN缓存身份、权重保存重载输出精确一致；smoke权重标记discarded，不能用于train/assess。`train`只接受完整且无failure的ideal/smoke报告，重新初始化参考；中断可从本版完整epoch边界用`--resume`恢复到新目录，不能续用smoke。`assess`只接受完成的固定epoch_04训练包，在432张留出和全部887张VAL读取真实图像，评分沿用SHA锁定原B框/score/漏检缓存，不声明本次在线B重现了旧缓存。

评价报告分别保留原输出覆盖、输出帧中心正确率、全帧中心正确覆盖、参考可用/不可用及其正确/错误支持、参考尺寸误差、真实尺寸错误类AUROC/AP、同接受数混淆与score同分边界、合成长/短边双侧辨识。AUROC主比较限定共同参考可用输出，另报score/simple的全部输出结果及不可用正确/错误数，不能掩盖参考可用性筛选。旧ROI只复用已核验VAL缓存，属于历史对照，不能称与新分支相同留出训练职责。没有部署门限、新三个标志替换或自动收益PASS。训练的成功终态只代表工程完成。

**本地核验（事实，服务器待运行）：** 21项必要unittest中19项CPU通过，2项实际Torch梯度/loss和保存重载检查因本地无Torch明确跳过，服务器环境应执行这两项。CPU包含独立截断Gaussian方差积分、尺度/反射还原、宽高/π等价、长短边双侧与中心/方向控制、不可用/漏检分母、同数score边界、阶段完成/失败保护、二进制流保存/角色校验，以及模拟完整assess文件输出的回归；模拟流程不是CUDA通过。Python3.8语法与来源合同已核验。当前版本[本地理想参考报告](../work_dirs/port_size_reference_v1_ideal_local_20261004_final_v3/ideal_report.json)的14个拟合角色原尺度视图全部通过，最大尺寸相对误差0.6941%，四类尺寸扰动的最小风险差0.668276；半尺度反射视图8/14通过，其余6图因短边模型分辨率不足不可用，不放宽16px保护门槛。对应[代码与证据复核记录](../work_dirs/port_size_reference_v1_code_review_local_20261004/review.json)保存源码/报告SHA、测试和真实错误支持。理想结果仅证明解析读取在标注派生热图上的数值机制成立；真实参考学习、实际显存/延迟和可靠性改善全部待服务器。TEST已多次暴露，本版没有TEST入口，不参与门限/权重/方法选择。

**服务器顺序（所有输出均新建目录，不覆盖旧证据）：**

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -m unittest discover -s tests -p 'test_port_size_reference_v1.py' -v

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_reference_v1.py \
  --mode ideal --out-dir work_dirs/port_size_reference_v1_ideal

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_reference_v1.py \
  --mode smoke --gpu 0 \
  --ideal-report work_dirs/port_size_reference_v1_ideal/ideal_report.json \
  --out-dir work_dirs/port_size_reference_v1_smoke

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_reference_v1.py \
  --mode train --gpu 0 \
  --ideal-report work_dirs/port_size_reference_v1_ideal/ideal_report.json \
  --smoke-report work_dirs/port_size_reference_v1_smoke/train_report.json \
  --out-dir work_dirs/port_size_reference_v1_train

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_reference_v1.py \
  --mode assess --gpu 0 \
  --reference-checkpoint work_dirs/port_size_reference_v1_train/epoch_04.pth \
  --out-dir work_dirs/port_size_reference_v1_assess
```

只有先得到`IDEAL_SIZE_READER_PASS`及`SIZE_REFERENCE_SMOKE_SAVE_RELOAD_PASS_DISCARDED`才能继续4轮有限训练。源码/原输入身份或数值门槛失败时先回传日志，不修改旧来源清单、重拟合simple、重建缓存、跳帧或放宽规则。默认复用`work_dirs/port_reliability_train_support_v1_cache`、`work_dirs/port_reliability_train_support_v1/train_input_snapshot.json`、`work_dirs/port_reliability_branches_v1_val_cached_v1`及`work_dirs/port_simple_reliability_v1_fit/policy.json`；目录若已改名，用对应参数指定同SHA文件。

回传ideal/smoke/train各自`completion.json`与主报告、assess的`assessment.json`、`reference_rows.jsonl`、`probe_rows.jsonl`、`completion.json`，训练轨迹为`train_steps.jsonl`；不必回传四轮权重。达到这些工程终态后再判断真实参考精度、双侧辨识、分域/逐视频的同数质量与可用覆盖是否支持正式实验；本版1536步是有限可行性预算，不能称最终正式方案训练已获科学放行。

## 11. 二维尺寸参考v1服务器结果与下一步（2026-10-04）

**证据与核验（事实）：** 用户提供`port_size_reference_v1_review_20261004_134359.tar.gz`及四阶段终端输出。结果保存在[服务器回传目录](../work_dirs/port_size_reference_v1_server_review_20261004_134359/)，主表为[assessment.json](../work_dirs/port_size_reference_v1_server_review_20261004_134359/port_size_reference_v1_assess/assessment.json)，本地[独立复核记录](../work_dirs/port_size_reference_v1_server_review_20261004_134359/review_analysis.json)保存包SHA、身份检查与补充偏差统计。四阶段合同一致；完成清单中17个随包文件SHA均通过，5个权重按打包要求排除，仅能核对声明SHA及训练/评价的epoch04身份一致，未本地重载权重。来源及父合同与当前代码、既有TRAIN/VAL/policy/旧ROI证据一致；1319条评价记录（432留出＋887 VAL）的原GT/B框/score/漏检与原缓存精确相同。四方法逐组AUROC以独立正负样本成对比较复算，同接受数错误接受全部一致；1318条探针重放最大风险浮点差1.11e−16。包与终端未提供服务器unittest日志，不补称全部单测已在服务器通过。本轮没有连接服务器、改算法、训练或读取TEST。

**工程结果（事实）：** ideal原尺度14/14通过，半尺度8/14通过、6图分辨率不可用；smoke四步通过实际梯度、B原输出侧计算前后精确一致及保存/重载检查。正式有限训练严格384张×4轮=1536步，全部拟合帧职责正确，无留出训练帧；各轮均loss为0.090314、0.033457、0.029543、0.025038，最大裁剪前梯度7.1904，零裁剪步。B状态摘要训练前后相同，参考epoch04与评价声明SHA一致。训练PyTorch峰值allocated364.803MiB、reserved484MiB；这不等于nvidia-smi总占用或其他任务共享占用。评价完成432＋887帧，参考对全部1318个B输出数值可用，1个VAL漏检单独保留。

**B覆盖保持（事实，缓存评分职责）：** real VAL输出374/375=99.7333%，输出帧中心正确360/374=96.2567%，全帧中心正确360/375=96.0000%；sim为512/512且全部中心正确。没有更改B框/score、删除中心或修复漏检；该评价不宣称完整在线B缓存复现或深度精度保证。

**尺寸参考不合格（事实）：** 参考相对GT的误差定义为规范长/短边最大相对误差。留出432图没有一图两边同时≤10%，平均最大相对误差92.6932%；VAL886输出也为0/886，real平均132.4750%、sim平均122.3230%。VAL参考短边/GT短边中位数real1.99277、sim2.29905；real374/374、sim508/512的短边高估超过10%。长边中位数real1.54078、sim1.20555，不符合单一统一尺度系数偏差的特征。数值可用保护未把这些几何偏大的参考标成不可用，故`defined=True`不能理解为尺寸准确或分量可信。

| VAL域 | 错误/输出 | 新参考AUROC/AP | score AUROC/AP | simple AUROC/AP | 旧ROI AUROC/AP |
|---|---:|---:|---:|---:|---:|
| real | 181/374 | 0.569748/0.573159 | 0.686285/0.660353 | 0.589500/0.603345 | 0.425615/0.476814 |
| sim | 23/512 | 0.453276/0.049428 | 0.711390/0.160515 | 0.740464/0.207551 | 0.637237/0.162190 |

错误为正类、风险越高越倾向错误；面积不是当前三标志的准确率。旧ROI属于历史不同训练职责对照。新参考在real/sim均未优于score；混合VAL新参考AUROC0.496377，simple混合AUROC0.831788不能掩盖其real及逐视频局限。新参考逐视频real_seq07 AUROC0.336149、real_seq14为0.535252。留出混合AUROC0.667640高于score0.582944，但只有4个真实错误，且参考尺寸全部不合格，不能用它放行。

在预设95%全帧名义接受点（向上取整），real同接受357框时新参考错误接受169、score166、simple164、旧ROI170；sim同接受487框时分别22、20、19、16。90%点real同接受338框分别159、153、155、168；sim同接受461框分别21、17、17、15。真实错误接受与正确误拒都必须报告，不把这些离线同数点称为已部署门限。

长/短边±15%双侧探针：留出432图的长边与短边双侧成功均0/432；VAL均0/886。单侧风险增加分别597/1728、937/3544，不能称双侧候选辨识成立。当前尺寸风险接近1（VAL real中位0.999191、sim0.999823），这是相对偏大参考的不一致程度，不是错误概率。float64风险仍可区分全部886输出，调整单调风险映射系数本身不会增加排序信息。

**解释与限制（推断，未定根因）：** 已证实当前“真实预测热图→二阶矩参考→尺寸一致性”整体没有通过，而理想标注热图的解析读取通过。当前最直接障碍是参考短边普遍高估；仅有loss下降不能证明范围学准。源码采用背景位置p²加权BCE，p很小时该项约按p³下降；二阶矩按距离平方累积非目标响应，存在训练像素损失与尺寸读取敏感度不一致的机制风险。广泛低响应、平滑/多峰、P3空间分辨率、模型/训练预算及跨视频泛化都仍可能参与。本包未保存完整预测热图、逐位置loss或范围质量贡献，不能据当前统计断言背景尾部是主因、方向结构已被证明无效或整个PQA思路无效；本版只是自定义有限改编。

**最推荐下一步（待授权实现）：** 不增加训练轮数或换权重，先用固定epoch04补一次TRAIN热图—读取诊断。预先按拟合五个序列各首尾选10图，加real_seq13与sim留出段各首尾4图；不按错误/效果挑帧，不新增VAL/TEST读图或调门限。保存原始预测图/GT几何目标/变换metadata/叠图，分报拟合与留出的参考尺寸、中心、峰/背景、GT范围内外质量与对二阶矩的贡献，GT范围仅离线诊断。由此区分拟合内范围没学准、读取对尾部敏感和新增分支泛化不足。若拟合图核心定位/范围已准但低响应显著放大矩，优先设计抗尾部的几何模板拟合读取及数值可用性；若核心范围也不准，再针对参考监督/分辨率设计一个限定版本。方法修改仅由拟合TRAIN证据制定，留出与全部VAL用于固定后评价。当前v1不替换simple、不扩展方向/中心、不放行大规模正式对照或新TEST。大小论文和独立几何交接均未修改。

## 12. 正式midpoint迁移与固定TRAIN读取改进（2026-10-04）

**授权及事实：** 用户确认检测前端已为B＋midpoint，授权按推荐迁移可靠性。本轮新增`run_port_midpoint_reliability_v1.py`、`port_midpoint_reliability_v1.py`、同名protocol/sources和单测，复用既有实现，没有修改检测/参考训练或旧来源清单。固定B epoch24与正式完整VAL选定`head_epoch_23.pth`（SHA `2c4c5ae9e071cbdb13fb722f4b2d3986861c62bd60c53568c72fb9b02d662f0b`），不加载短拟合或按最新文件改选。formal `selection.json`、24轮VAL选权证据、实际权重字节、原ROI缓存和原TRAIN/VAL证据均校验身份。第11节“待授权”由本次授权取代，但失败结果不改写。

**实现与比较：** `collect`逐份读取已有`train_s1.pt`/`val_s1.pt`，只前向冻结midpoint，不提取新图像/P3、不加载半尺度缓存。保留全部2558 TRAIN/887 VAL、完整回退、B输出数量/score和原资格；VAL重放须与正式选择一致（框容差1e−4/1e−6、score精确相同），评分保留已核验正式VAL数值并报重放差。`fit`沿用三特征、正负类平衡Newton/L2=.1及95% pooled VAL覆盖规则，重新拟合尺寸/方向参数与门限，输出三个标志。中心继续保留有效最终框，不新训练中心预测器。报告配对正式B＋旧policy、midpoint＋旧policy、midpoint＋新policy，分域/逐视频、同数score错误接受/正确误拒与错误类AUROC/AP。中心正确<15px仅统计输出帧，另报输出覆盖与全帧正确覆盖。sim监督仍为Webots OBB，无需轴线。

**固定读取候选（待真实证据）：** `probe`复用尺寸参考epoch04（SHA `72b621fe2f0b78ac5292f2142eb318ab256791ab87ce54eee75e50f217e0673a`）及共享冻结B P3。在既有参考划分中按每序列首尾固定10张拟合＋4张留出TRAIN；不按GT误差挑帧。两个读取器共用最终midpoint框上下文：旧二阶矩，与新增主峰连通半峰核心的robust log-Gaussian模板拟合。固定半峰=.5、至少16核心单元、IRLS10次/Huber=.15、log-RMSE≤.15、condition≤1e8、原模型短边≥16px及边界保护；不搜门限。协方差由拟合曲率还原，长短边=4σ，不再套截断矩校正。上下文仍依赖最终框中心/长边；只改中心/方向的固定参考探针不代表完整链解耦。

保存预测/目标热图、变换metadata、同尺度PNG及GT范围外响应质量与长/短轴二阶矩贡献，分报拟合/留出、两方法可用性及共同可用集合误差、长短边双侧辨识。GT只进入离线诊断及理想数值检查；新读取器不自动替换simple标志、不被称为已校准错误概率。拟合内核心范围也不准时，继续改读取不能认定解决问题；本次真实热图结果决定是否值得后续学习改进。

**影响解释（事实与推断）：** 几何修正会改变最终框、simple特征及正确/错误标签；所以原B可靠性参数不能直接视为新前端性能，几何改善也不保证风险排序/状态准确率改善。冻结B P3未改变，允许复用参考权重；这不保证新上下文下有效。可靠性API不接受GT、不写回任何最终OBB/score/数量，`smoke`检查6张固定VAL在线输出及模型状态前后相同。等比例变换/原图坐标还原沿用原链，接口不变不等于深度精度保证。参考留出仅针对epoch04分支；新simple拟合全部TRAIN，不能把那4图称为对整条新管线的独立留出。

**本地核验：** 22项新增单测中21项CPU通过，1项实际Torch适配/空输出状态检查因本地无Torch跳过，服务器须执行。已验证来源合同、Python3.8语法、CLI、完整模拟collect/fit输出与重载、锁定VAL数值保护，以及真实既有TRAIN/VAL来源身份。固定10张拟合角色的标注派生理想热图均通过，最大尺寸相对误差0.03854%；仅为解析数值证据，未运行真实epoch04/midpoint权重、GPU前向、新simple拟合或TEST。服务器显存/速度与方法效果待运行，不承诺零性能下降。

**服务器顺序：** 上传新增5文件，并确认依赖`analyze_port_geometry_midpoint_size_temporal_v1.py`与本地版本相同。项目根目录运行本版unittest，再依次`check → probe → collect → fit → smoke`；probe仅产生待审证据，其工程完成不自动批准参考部署。collect/fit迁移不依赖参考效果通过。每阶段使用新目录，来源不一致时停止，不能更新旧manifest绕过、重建B缓存或跳帧。只复用`work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json`、`work_dirs/port_geometry_midpoint_formal_v1_roi_cache`及原参考/简单policy输入。此次不训练检测/参考/ROI、不修改大小论文；TEST已多次暴露，本版没有TEST入口，也不以其选择权重/规则。服务器输出留在服务器，本地不新增回传结果副本或复核报告目录。

## 13. midpoint迁移服务器运行与probe接口修复（2026-10-04）

**终端事实，完整结果待回传：** 用户提供终端记录，原22项服务器单测全部通过（含本地跳过的实际Torch适配/空输出测试），`check`、2558 TRAIN＋887 VAL的`collect`、`fit`与6帧`smoke`均得到相应成功终态。real最终框中心正确362/374=96.7914%，全帧362/375=96.5333%，输出覆盖374/375=99.7333%；sim为512/512且中心全部正确。这是正式midpoint框的中心结果，不证明新尺寸/方向可靠性优于旧参数或score。尚未收到fit_report/policy/逐帧完整文件，不能补称已复核全部收益或源文件SHA。本轮没有TEST。

**实现缺陷及修复（事实）：** `probe`首帧在`offline_map_evidence`把完整六维`[cx,cy,w,h,angle,score]`传给只接受五维的`canonical`，报`Expected a finite positive OBB`；此前两个读取器接收`pred[:5]`已正常执行，报错不表明预测框尺寸非法。离线诊断现明确接受5维几何或6维有效检测；6维先验证原score再仅提取5维几何，score不参与矩、输入不被修改。补齐实际probe调用回归后还定位到原数值检查的NumPy布尔/整数无法JSON序列化；已将两处相对误差转换为Python float，令门槛布尔与汇总计数为原生类型。未改变模板公式、参数、分母、模型权重或simple决策。

**复核与续跑：** 新增六维/五维诊断等价、非法score拒绝，以及模拟完整14帧probe实际保存NPZ/PNG/逐帧/汇总报告的回归。24项检查中23项本地CPU通过，1项Torch因本地无依赖跳过；新完整probe是模拟调用证据，真实热图仍待服务器。更新本版sources中runner/helper/tests的SHA；没有改父协议/清单，也不重写旧成功completion。上传4个变化文件（runner、helper、tests、sources），执行新版单测，仅重跑`probe`到`work_dirs/port_midpoint_reliability_v1_probe_fix_v1`，保留失败目录。原check/collect/fit/smoke不因这个诊断修复要求重算；结果仍按原源码合同保存，新增probe按修复后的源码合同保存，后续复核明确两个版本并分别校验，不能混写或改旧合同来消除差异。

## 14. midpoint可靠性回传复核与结论（2026-10-04）

**包及身份（事实）：** 直接在内存读取用户提供`port_midpoint_reliability_v1_review_20261004_150523.tar.gz`，包SHA `5e32f9a045fbd3f538d56ae35a15ca8eeeb904557bf69ce554a206974719943e`；46文件包含5个成功阶段、14组NPZ/PNG，未含failure或未修复probe。41个完成清单文件SHA全匹配，各input_check与completion合同相同，检测/midpoint/参考更新为0且TEST未读。check/collect/fit/smoke合同一致；新probe仅sources不同，三个变化源码与此前接口/JSON修复一致。原源码SHA逐项对照Git `3f3afc5`通过，新probe对照当前源码/清单通过；前端均为B24＋正式midpoint23，参考仍epoch04。权重、原正式ROI缓存及完整24轮原选权文件未包含，本地未重新加载权重，不把声明SHA当成本次字节复核；前次服务器check已核验。

**独立复算（事实）：** 2558 TRAIN＋887 VAL的GT、图片尺寸、角色及real原生/Webots方向资格与既有核验源逐项精确相同，14图选择也与预先固定首尾/参考拟合留出一致。VAL底层记录指纹匹配正式选定结果，缓存重放报告最大框差0。重新计算两个TRAIN风险模型及pooled VAL覆盖门限，数值在1e−10内相同；三个对照的分域/逐视频、错误接受/正确误拒、完整OBB与887条导出决策均重现。AUROC另以正负样本成对比较复核。14份原概率图/变换metadata重建GT目标与valid精确相同，两读取器、合成扰动与框外贡献均复算相同；查看拟合正常、拟合偏差与留出偏差PNG。6帧smoke输出与固定VAL/新policy相符，模型状态前后相同。未创建本地回传目录/复核结果副本；本包不含修复后24项单测日志，不补称其全部已在服务器通过。

**几何变化（事实，固定VAL，不重选前端）：** real输出仍374/375=99.7333%，中心360→362/374（96.2567%→96.7914%），全帧96.0000%→96.5333%；sim均512/512及100%中心正确。尺寸错误real181→188/374、sim23→56/512；方向错误real67→61/374、sim69→34/512。完整OBB raw共同正确606→600/887。TRAIN尺寸错误real54→46、sim2→0，方向按评价资格real191→96、sim29→0；新模型拟合全部TRAIN方向资格的错误为real153、sim0。由此确认前端改变了真实标签分布，中心/方向收益不等于尺寸收益，TRAIN错误减少不保证VAL减少。

**simple重新拟合（事实）：** 尺寸46坏/2512好，方向153坏/2405好，均5次Newton收敛；所有坏样本来自real。新两个分量在混合VAL分别接受843/887=95.0395%，并非各域/完整OBB均95%。新size门限.3977338761，angle .4143115365；score风险门限均.7043236196。与旧参数直接迁移相比，新size/angle错误接受混合210/81→212/80，接受数839/842→843/843，数量不同不能仅按错误接受变化判优劣。

| 最终midpoint VAL域/分量 | simple接受 | simple错误接受 | 同接受数score错误接受 | simple正确误拒 | simple输出帧状态准确率 |
|---|---:|---:|---:|---:|---:|
| real/size | 333 | 158 | 159 | 11 | 54.8128% |
| real/angle | 334 | 48 | 44 | 27 | 79.9465% |
| sim/size | 510 | 54 | 54 | 0 | 89.4531% |
| sim/angle | 509 | 32 | 33 | 1 | 93.5547% |

状态准确率只以该分量可评输出为分母，错误接受＋正确误拒为状态误判；漏检另报。real新simple错误类AUROC size.593428/angle.635992，score.710278/.711203；sim为.700149/.652781对score.630013/.618385。逐视频real_seq07尺寸simple.585510优于score.443998，同数196接受错误154对160；real_seq14同数137时4对6。分域与逐视频排序不是相同接受分配，不能据混合结果声称全部视频胜负。新simple完整OBB接受839/887=94.5885%，共同正确589/887=66.4036%；raw600/887=67.6437%。相较score没有稳定、显著或统一收益；此为source TRAIN拟合/反复使用source VAL的描述性证据，未新增TEST。

**模板读取改善（事实，尚未替换simple）：** 10拟合＋4参考留出图两方法全部数值可用。拟合平均最大长短边相对误差：二阶矩91.63395%→模板12.51171%，10%内0→6/10；留出111.18567%→19.29295%，10%内仍0/4。混合97.22016%→14.44921%，14张中11张模板两边均偏大、1张两边偏小、2张长短边混合偏差；最大误差30.52%，不能称尺寸合格。原max聚合风险长边双侧0→4/10、短边0→5/10；留出长短均0/4。补看不聚合的单边绝对log误差，拟合长/短双侧5/6、留出1/0：max还会掩盖非主导边变化，但剩余参考偏差也确实存在，不能只通过换风险映射解释失败。14组核心单元全在GT OBB内，不等于曲率/恢复尺寸正确；log拟合残差小也不等于符合GT尺寸。

**机制证据与限制：** 14图GT框外响应质量占比中位21.7995%（13.26%—31.64%），却贡献短轴二阶矩中位77.5257%（50.79%—85.28%）；固定同图/同参考权重仅改读取后误差明显下降，支持“尾部放大矩是v1尺寸偏大的重要因素”。框外以GT OBB代理界定，不等于已验证真实背景掩码；这是固定14图局部机制，不能泛称唯一根因。拟合real_seq06尾帧模板误差30.52%、留出real_seq13首帧28.83%且峰中心误差24.87px，显示非Gaussian核心/位置及跨视频范围学习仍有问题。新simple只用框级三特征，TRAIN bad稀少且sim无bad、VAL新增sim size56个，是错误支持及误差规律转移不足的证据；“主要靠域/视频代理而非局部质量”仍为推断，不能当已证明根因。

**下一步建议（已由第15节授权实现）：** 此次工程正确、fixed midpoint三标志接口已完成；不将其称为可靠性性能优化成功，不因本包改选midpoint或用TEST调参。保留固定参考epoch04、模板读取参数和最终midpoint上下文，下一步仅将这个已经实现的尺寸候选扩展到既有432参考留出TRAIN＋全部887 VAL，比较同一最终框上的score/simple/旧矩/新模板，分域/逐视频、共同可用集合/不可用帧、同数错误接受及全帧覆盖完整报告。这是固定候选效果验证，非再搜索根因/权重/阈值，也不重新训练；14图改善不足以代替全量可分性收益。若全量仍不支持参考有效，再设计一项范围明确的核心范围监督训练改进，而不是继续拟合相同三特征或直接加入方向结构。全部结果保持服务器，本地仅记录本节。

## 15. 固定midpoint基线与尺寸模板全量验证入口（2026-10-04）

**授权与范围（事实）：** 用户明确固定B24＋正式midpoint23为后续基线，授权按第14节建议固定读取、完成一次全量效果验证。新增[运行入口](../crane_project/tools/eval_port_midpoint_size_template_v1.py)、[评价工具](../crane_project/utils/port_midpoint_size_template_v1.py)、[协议](../crane_project/tools/port_midpoint_size_template_v1_protocol.json)、[来源清单](../crane_project/tools/port_midpoint_size_template_v1_sources.json)及[CPU回归](../tests/test_port_midpoint_size_template_v1.py)。原midpoint、参考、simple及几何代码/协议未改；没有启动训练、连接服务器或更改大小论文。本次只更新可靠性交接，没有改独立几何交接；其已有未提交修改属于另一个窗口。

**固定身份：** 复用原B epoch24、正式`head_epoch_23.pth`、参考`epoch_04.pth`和第14节已重新拟合的midpoint simple policy。模板参数仍半峰.5、至少16单元、10次IRLS/Huber .15、log-RMSE≤.15、condition≤1e8；上下文/弱证据/分辨率/边界保护及risk_scale=.1均继承原协议。不读取半尺度ROI、不重新collect/fit、不按新结果选择权重、系数、阈值或序列。check为CPU来源校验，run为固定GPU前向；无TEST/训练/新部署门限入口。

**旧产物复用的来源约束：** 原collect/fit成功阶段在probe接口修复前生成，合同`7afc84669a0144c42de0c28a452aaf111fc54129b301614df862a5603500d20f`与当前migration只差sources。新增协议固定该已审合同、两个completion字节SHA及全部产物SHA；允许来源版本差异只限这一确切已审产物，其他前端、数据、正式选权证明、参考合同、协议必须与当前prepare精确相同。当前源码仍按现有父manifest验证；不执行历史附件中的指令、不重写旧completion或放宽数据/权重检查。原3445行集合和policy不被覆盖。

**全量设计与事实支持：** 原参考划分的432留出TRAIN（real_seq13 304＋sim_seq08留出128）及全部887 VAL（real375＋sim512），共1319帧，包含原1个VAL漏检。留出只针对epoch04参考学习，B与simple已见过整个TRAIN；VAL也已经用于前端选择/工作点及多次开发，二者均不能称整条方法的独立确认。直接在内存重读第14节旧包核对计划、来源及字节pin；未解包到新本地目录。锁定midpoint框的尺寸错误：参考留出real4/sim0，VAL real188/sim56。留出错误太少，尤其sim单类AUROC为空，不能以高正确率证明错误辨识。

**评价与在线保持：** batch1逐图提取共享冻结P3，`no_grad`、模型eval/禁止梯度，不累积GPU热图/特征；只保存数值JSONL，不新增全量PNG/NPZ。每图复核图片SHA、原图坐标链，在线B及最终midpoint须与正式集合一致（几何atol1e−4/rtol1e−6、score精确相同）；参考侧计算前后输出须完全相同，结束核对三模型状态、policy和来源。评分/读取使用同一锁定最终框，GT仅在完成在线读数后生成离线正确性标签/参考尺寸误差；没有用GT生成在线参考/接受规则。等比例变换及深度接口约束保持，不据此宣称深度精度保证。

报告按两个角色分别分域/逐视频给出：中心命中率仅统计输出帧，另报输出覆盖、全帧正确中心覆盖；冻结三个标志及完整OBB的已有统计；尺寸原始错误、参考相对GT误差和10%内数量；四评分score/simple/旧矩/新模板的错误类AUROC/AP。在两读取器共同可用集合做相同支持的四方比较，并另报各读取器自身可用集合、不可用好/坏样本/原因与原缺输出。90%/95%是预先固定的全帧名义数量诊断，各方法以相同接受数比较，同时给全部方法并列分数的错误接受上下界。另列全输出统计，将参考不可用视作尺寸拒绝的假设、score/simple可使用全部输出；它与共同支持比较分开，不把不可用样本隐藏或当成尺寸正确。所有新参考接受点均为离线诊断，未创建模板在线标志/部署阈值，中心不因尺寸拒绝而删除。

**本地验证与边界：** 12项新版CPU单测全部通过，覆盖真实解析读取、GT不影响风险、缺输出/不可用/共同集合/全输出分母、并列分数与标签无关选择、正式在线容差/score精确保护、1319帧计划及参考fit/guard隔离、历史完成字节pin/失败拒绝、完整模拟逐帧运行与JSON保存重放/完成清单。新增代码按Python3.8语法核对，CLI可调用，来源清单通过；服务器真实CUDA/权重运行未在本地执行。GPU显存由单图流式处理控制，不承诺其他进程共享环境下总占用；运行后报告allocated/reserved峰值。

**服务器运行：** 上传新增5个代码/协议/测试文件；交接文件是第6个文档，可同步。保留现有已修复的midpoint入口/helper/sources，以及所有父依赖；不要重新生成旧来源清单。先运行新增单测与check，成功后run，一律新输出目录：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -m unittest discover -s tests -p 'test_port_midpoint_size_template_v1.py' -v

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_midpoint_size_template_v1.py \
  --mode check \
  --out-dir work_dirs/port_midpoint_size_template_v1_check

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_midpoint_size_template_v1.py \
  --mode run --gpu 0 \
  --out-dir work_dirs/port_midpoint_size_template_v1_assess
```

成功终态应为`FIXED_MIDPOINT_SIZE_TEMPLATE_FULL_ASSESSMENT_COMPLETE_REVIEW_REQUIRED`，工程完成不等于性能通过。待回传重点看参考尺寸是否改善、分域/逐视频共同可用排序是否优于score/simple，以及相同数量下错误接受/正确误拒和不可用带来的覆盖代价；混合AUC或单个视频提升不能独自放行。若不支持有效，再限定到参考核心范围学习的一项训练改进；本轮不追加诊断搜索/训练/方向分支/TEST。已暴露TEST不作权重/阈值/结构选择，旧B TEST保持旧身份。结果仍留服务器，本地不新建失败复核目录。

## 16. 固定模板全量服务器结果与基线比较（2026-10-04）

**运行与核验（事实）：** 读取用户终端及`port_midpoint_size_template_v1_review_20261004_161835.tar.gz`，包SHA `900f5ae29baba9a7b93a1754732184bf21d0654697d27336b5301c67dbb1c86a`、939435字节，2目录/7文件、无失败或路径/链接异常。终端12项新增服务器单测全部通过；check通过，run终态`FIXED_MIDPOINT_SIZE_TEMPLATE_FULL_ASSESSMENT_COMPLETE_REVIEW_REQUIRED`。五个完成清单产物SHA全部匹配；check/run/report合同一致，与当前源码/协议及父来源精确匹配。实际1319行=432参考留出TRAIN＋887 VAL，逐项比对第14节已审原集合，GT/最终框/B框/score/角色/缺输出完全相同。冻结policy SHA仍`1adac499369bade5ef1f79495260002697d2b698c80ebf5424d3d77438fae4af`，三个标志重放完全相同，风险末位最大差1.11e−16。两角色逐组summary及冻结三标志统计复算完全相同，另以正负样本成对比较复核66组AUROC、以独立阈值计数复核AP、标量复核共同可用同数错误接受/正确误拒；参考GT误差与log比例风险另算最大差1.11e−16。本地仅在内存读取，不新增回传目录/结果副本，未重跑训练或连接服务器。

**模型保持（事实）：** B24＋正式midpoint23、参考04和新simple与第14节身份一致，检测/头/参考/policy更新均0。服务器报告1319帧在线B/最终midpoint对锁定集合的最大绝对框差均0，每帧参考侧计算前后输出精确相同，三模型状态前后相同；PyTorch峰值allocated343.478MiB/reserved412MiB，不等于nvidia-smi总占用。回传无权重/热图，本地未重载GPU或重算模板拟合；能复核逐帧原框、参考读数、风险、数值身份/统计，不能把此数值复核称为本地全量图像重跑。

**当前基线与上轮（事实）：** 本轮没有改训练/工作点，固定midpoint＋simple的结果与第14节一致。VAL real输出374/375=99.7333%，中心正确362/374=96.7914%，全帧362/375=96.5333%；sim512/512及全部中心正确。相较原B，real中心360→362、尺寸错误181→188、方向错误67→61；sim尺寸错误23→56、方向错误69→34。这是同VAL配对几何变化，不把旧B TEST当成新midpoint TEST。simple固定尺寸real接受333/374、错误接受158、正确误拒11，方向334/374、错误接受48、正确误拒27；sim尺寸510/512、错误接受54/误拒0，方向509/512、错误接受32/误拒1。整体simple完整OBB接受839/887、共同正确589/887；raw共同正确600/887。模板仅候选离线评分，未产生新在线标志或修正几何。

**读取范围改善（事实）：** 误差是参考相对GT规范长短边的最大相对误差，不是检测误差或状态准确率。

| 共同可用角色/域 | 二阶矩平均误差 | 模板平均误差 | 模板两边均10%内/数值可用 |
|---|---:|---:|---:|
| 参考留出TRAIN整体 | 91.9224% | 21.4866% | 67/432 |
| VAL real | 131.1518% | 19.1554% | 72/373 |
| VAL sim | 121.6411% | 17.3256% | 197/512 |

旧矩所有角色两边均10%内为0。模板参考留出real23/304、sim44/128；VAL整体269/885=30.3955%，仍有616/885两边不能同时合格。模板real的最大误差369.27%，sim162.88%，不能仅凭均值改善视为可信尺寸参考。模板对原缺输出保持不可用，另有一个real坏尺寸输出`real_seq07_00018`因`core_clipped`不可用：real共同支持373（187坏/186好），sim512（56坏/456好）。数值可用不等于几何正确。

**同一最终框的尺寸错误可分性（事实）：** 以下错误为正类，使用两读取器共同可用集合；AUROC不是状态准确率。

| VAL域 | score AUROC/AP | simple AUROC/AP | 旧矩 AUROC/AP | 模板 AUROC/AP |
|---|---:|---:|---:|---:|
| real（373输出） | .708786/.683922 | .591283/.610948 | .559686/.593473 | .360388/.445744 |
| sim（512输出） | .630013/.215799 | .700149/.285911 | .492011/.147529 | .682253/.311814 |

real模板低于旧矩、score和simple；sim模板优于旧矩/score的AUROC，低于simple的AUROC但AP略高。逐视频模板real_seq07 AUROC .416793、real_seq14 .291751，均未优于该视频simple .583207/.753521。混合VAL模板AUROC .478238、simple .783515不能替代分域判断。参考留出real仅4坏样本，模板AUROC .708333与score .699167接近；95%接受289时模板仍接受全部4坏、score3坏。留出sim没有坏样本，AUROC/AP为空；不得以100%多数正确样本证明检错成立。

**同数接受与误判代价（事实）：** 95%全帧名义数量为real357、sim487；下表为全输出对照，参考不可用视为拒绝，score/simple可使用所有输出。每格为“错误接受/正确误拒”，原漏检不混入状态分母。并列边界上下界在这两组均相等，不改变胜负。

| VAL域/接受数 | score | simple | 旧矩 | 模板 |
|---|---:|---:|---:|---:|
| real/357 | 173/2 | 172/1 | 175/4 | 176/5 |
| sim/487 | 50/19 | 46/15 | 49/18 | 42/11 |

real模板比simple多接受4坏、多误拒4好，尺寸状态准确率51.6043%对53.7433%；sim少接受4坏、少误拒4好，89.6484%对88.0859%。共同支持下real score是172/1，因为剔除1个模板不可用坏框；其余三方法数字相同。不能混用共同集合172与全输出173。90%共同数量real338时模板167坏/15误拒，对simple161/9；sim461时模板40/35，对simple43/38。sim收益是固定离线数量点，不能称新部署阈值已通过，更不能据域/视频选择方法。

**可观察问题及推断边界：** real共同支持中的好检测框，其模板短边/GT中位1.14639，模板风险中位.78902；坏检测框预测长/短边比GT中位.88071/.87434，模板长/短边比GT中位.92814/.90371，风险反而中位.67509。已观察到参考与预测存在同向偏小、好框参考偏大的分组特征，符合“两个估计彼此相近却不接近GT”的失败机制；分组中位数不证明每帧的误差相关性、唯一根因或图像证据天然无效。sim坏框模板风险中位.85633高于好框.74457，符合该域较有效的排序。抗尾部读取修复了v1的大量尺寸高估，但参考核心的范围/跨视频尺度学习仍不足；错误含义从原矩统计偏差转向模板仍不够准及参考—检测的一致性与正确性不等价，不能靠统一风险反号/域特定阈值在VAL上挽救。

**结论与下一步（建议，尚未实现）：** 工程与全量固定效果验证完成；继续固定midpoint前端和现有simple作为对照，模板不替换在线接口，不宣称可靠性优化成功。无需再重复同权重的全量诊断或先跑TEST寻找改善；当前TEST已反复暴露。下一项应限定为参考学习中的核心范围/长短边几何监督改进，保持B/midpoint、模板读取和评价协议固定，在TRAIN设计并按同预算对照VAL，目标是减少参考自身误差及与坏检测同时偏小的情形。不能只增加泛化训练轮数、继续三特征重拟合、继续搜读取系数或同时扩展方向结构。本节仅记录测量和建议，未修改代码/大小论文，亦未启动新训练。

## 17. 固定midpoint＋普通三标志的TEST评价入口（2026-10-04）

**授权与范围（事实）：** 用户授权按既定流程补齐TEST三分量评价。本轮新增[CPU入口](../crane_project/tools/eval_port_midpoint_simple_reliability_v1_test.py)、[冻结协议](../crane_project/tools/port_midpoint_simple_reliability_v1_test_protocol.json)、[来源清单](../crane_project/tools/port_midpoint_simple_reliability_v1_test_sources.json)及[必要回归](../tests/test_port_midpoint_simple_reliability_test_v1.py)，只更新本可靠性交接。没有修改原检测/midpoint/simple/参考代码、父来源清单、几何交接或大小论文，没有训练、连接服务器或新建本地复核结果目录。当前固定流程为`B epoch24＋正式midpoint epoch23 → 现有simple三个独立使用标志 → 分量质量与覆盖评价`；尺寸模板、ROI、结构分支均不参与，不把补齐入口称为可靠性优化成功。

**复用输入与身份（事实）：** 新入口读取几何窗口已有`work_dirs/port_geometry_midpoint_formal_v1_test_eval`的五个正式TEST文件；锁定完整1440帧、原图坐标、B24和`head_epoch_23.pth`，并检查正式选权证明、训练/评价来源、模型状态前后相同、更新数零和完成清单SHA。普通三标志使用第14节已审`work_dirs/port_midpoint_reliability_v1_fit/policy.json`，SHA仍`1adac499369bade5ef1f79495260002697d2b698c80ebf5424d3d77438fae4af`，连同原fit四产物及completion精确字节验证。该fit发生在probe接口修复前，按原已审合同`7afc84669a0144c42de0c28a452aaf111fc54129b301614df862a5603500d20f`锁定，不重写旧completion或manifest、不重新fit。原TRAIN参数及VAL门限均冻结。

正式TEST逐帧文件没有图像宽高，故从已有`work_dirs/port_reliability_branches_v1_test_cached_v1`补充图像尺寸、图像/标注身份及GT一致性；此缓存的ROI/结构质量和旧预测不进入新方法评分。旧B policy用于迁移对照。两侧预测均取同一次正式midpoint评价中的B框与最终框，避免用旧次B推理框代替同次对照。正式GT是本次几何数值的评价依据，和旧元数据GT以atol1e−4/rtol1e−6核对，保留原正式GT精度；不重新计算/重建标注。模型权重及图像字节不再加载，当前数据目录不重哈希，也不宣称本次重新验证全量在线推理一致性。

**评价内容（实现事实，效果待服务器）：** 按all、real/sim及三个序列分别报告`raw`、冻结门限`score_only`、冻结`simple`的中心/尺寸/方向接受与拒绝。中心仍是“有效最终框即保留”，不是新增中心正确性预测器；无输出三个标志均false，尺寸/方向拒绝不删除中心或修正最终框。在线判断仅接收最终OBB/score和原图宽高，GT仅离线生成正确性标签；规则为中心<15px、规范长短边最大相对误差≤10%、π周期长边角误差≤3°，GT aspect≥1.2仅限定离线方向资格。中心命中率只以输出帧为分母，另报输出覆盖率和全帧中心正确覆盖。

尺寸/方向明确给出`bad_accepted`（错误接受）、`good_false_rejected`（正确误拒）、`bad_correctly_rejected`（正确检出的坏框）、接受后正确率、输出帧状态准确率和全帧正确覆盖；状态误判=错误接受＋正确误拒，缺输出不混入状态准确率分母。另报全部帧标志覆盖、方向不可评接受、完整OBB和按真实帧号的连续可用性。匹配simple接受数的score只作离线诊断，边界并列分数报告错误接受上下界，不使用GT挑选同分框或产生TEST部署门限。配对B＋旧simple与midpoint＋新simple的比较同时包含几何与拟合参数迁移，不能独自证明方法创新。

**本地验证与限制（事实）：** 16项新增CPU检查全部通过，覆盖Python3.8语法及新进程不导入Torch/MMCV/MMRotate/OpenCV、固定权重/来源身份、错误阶段/权重/数据拒绝、完整帧和漏检、GT不进入三个标志、独立分量拒绝、宽高交换/角度周期、同数score并列诊断，以及完整模拟输出的JSONL/CSV/报告保存重载与完成清单/禁止覆盖。几何等价的浮点误差用数值容差比较，接受决策和整数计数精确一致。另在内存直接读取已提供正式TEST及midpoint迁移压缩包，结合原已审尺寸元数据，通过真实1440帧/1431输出输入检查及全部字节pin；没有解包或计算新TEST成绩。本地尚未运行服务器入口，正式新可靠性结果待回传，不能用输入检查或模拟测试代替性能结果。

**服务器指令：** 上传四个新增代码/协议/来源/测试文件；本交接可一并同步。保留现有父依赖和原fit/正式TEST/元数据目录。下列步骤都是CPU，无CUDA、训练或再次检测；每次使用新输出目录。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -m unittest discover -s tests \
  -p 'test_port_midpoint_simple_reliability_test_v1.py' -v

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_midpoint_simple_reliability_v1_test.py \
  --mode check \
  --out-dir work_dirs/port_midpoint_simple_reliability_v1_test_check

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_midpoint_simple_reliability_v1_test.py \
  --mode run \
  --out-dir work_dirs/port_midpoint_simple_reliability_v1_test_cached_v1
```

默认依赖：`--policy work_dirs/port_midpoint_reliability_v1_fit/policy.json`、`--old-policy work_dirs/port_simple_reliability_v1_fit/policy.json`、`--formal-test-dir work_dirs/port_geometry_midpoint_formal_v1_test_eval`、`--metadata-dir work_dirs/port_reliability_branches_v1_test_cached_v1`。目录若改名可显式传参，但必须是同SHA的完整原产物；身份失败时停止，不更改清单、跳帧、重拟合或重建检测缓存绕过。check成功终态为`FIXED_MIDPOINT_SIMPLE_TEST_INPUTS_PASS`，run为`FIXED_MIDPOINT_SIMPLE_TEST_COMPLETE_REVIEW_REQUIRED`；工程完成不代表性能通过。

回传check/run整个小结果目录即可，run主要文件为`test_compare.json`、`test_decisions.jsonl`、`component_metrics.csv`、`test_summary.md`、`input_check.json`与`completion.json`，无需权重/图像。后续按错误接受、正确误拒、覆盖与逐视频一致性判断，不依据TEST修改规则、权重或选择候选。**TEST已经多次暴露**；本次是冻结流程的报告，不称未接触的独立确认。等比例变换/原图还原沿用原链，保持接口不等于深度精度保证。

## 18. 固定midpoint＋simple的TEST回传及下一步（2026-10-04）

**工程与证据核验（事实）：** 用户提供`port_midpoint_simple_reliability_v1_test_review_20261004_190008.tar.gz`，600696字节，SHA `5c93f6fdc250aa38af2480b9d0102c5ed6e2a4d617f6e9ff390eef37c9d2f0d1`。包内2目录/8文件，无失败文件、重复身份、路径逃逸或链接。check终态`FIXED_MIDPOINT_SIMPLE_TEST_INPUTS_PASS`，run终态`FIXED_MIDPOINT_SIMPLE_TEST_COMPLETE_REVIEW_REQUIRED`；两份completion索引的6个产物SHA全部匹配。两阶段合同、来源及input_proof与当前受保护代码、原已审midpoint policy和正式几何TEST产物一致，B24/正式midpoint23/原fit policy均未更换，detector/midpoint/reliability/threshold更新及新检测推理均0，GPU_used=false。不从本包补称服务器执行了16项unittest：本次未附单测终端日志。整个包只在内存读取，不新增本地解包目录或结果副本，没有连接服务器或修改算法/大小论文/几何交接。

使用原正式TEST包与第14节已审fit包、原TEST图像尺寸元数据重新通过输入校验，共1440帧、1431输出，方向GT资格1440/1440；正式最终框与同次B框的GT/score/身份/缺输出状态保持。逐帧三个标志、全部汇总及配对B汇总重放一致，244个风险末位浮点差最大1.11e−16，未改变决策。独立标量计算规范长短边、π周期角误差和中心距离，对两前端×6组×3分量×4方法共144组混淆计数完全一致；CSV72行与JSON对应统计一致，报告数值指纹及summary文本重放一致。此为CPU数值/来源复核，不是重新GPU推理、权重字节重载或独立重复训练。

**中心及覆盖（事实）：** real输出859/868=98.9631%，中心正确858/859=99.8836%，全帧正确中心858/868=98.8479%；sim三项均572/572=100%。与同次B的中心正确/错误标签逐帧相同，不代表连续距离误差未改变。中心有输出即保留，1个real坏中心也被保留，尚不是独立中心正确性预测器；9个漏检仍单独报告。三个方法real中心最长不可用4帧；simple尺寸21帧、方向16帧，sim均0。标志连续可用不等于几何连续正确，长度按真实帧号统计，不伪造秒数。

**新前端固定三标志（事实）：** 正确性阈值沿用中心<15px、最大长短边相对误差≤10%、方向≤3°。下表分母“接受/总帧”体现缺输出与拒绝；状态准确率仅以输出帧为分母，状态误判=错误接受＋正确误拒。

| 域/分量 | simple接受/总帧 | 错误接受 | 正确误拒 | 正确检出的坏框 | 接受后正确率 | 输出帧状态准确率 | 全帧正确覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|
| real尺寸 | 737/868 | 289 | 32 | 90 | 60.7870% | 62.6310% | 51.6129% |
| real方向 | 781/868 | 145 | 39 | 39 | 81.4341% | 78.5797% | 73.2719% |
| sim尺寸 | 572/572 | 58 | 0 | 0 | 89.8601% | 89.8601% | 89.8601% |
| sim方向 | 572/572 | 6 | 0 | 0 | 98.9510% | 98.9510% | 98.9510% |

real尺寸有379坏/480好，simple拒绝90坏和32好，坏框检出率23.7467%；289错误接受不是全部状态误判，合计321/859。尺寸状态准确率由raw55.8789%至simple62.6310%，但正确尺寸全帧覆盖480/868→448/868。real方向原184坏/675好，simple拒绝39坏和39好，检出率21.1957%，状态准确率与raw均78.5797%，接受后正确率由78.5797%至81.4341%，正确方向全帧覆盖675/868→636/868。sim两分量全部接受，坏框检出率0；方向98.9510%主要来自前端原始方向正确，不能归功于可靠性筛选。

**公平的同数量比较（事实及限制）：** 每格为错误接受/正确误拒；score按对应分组匹配simple数量，是离线诊断，不是TEST选择的部署门限。表中边界并列上下界均相同。

| 分组/分量 | 共同接受数量 | simple | 同数score |
|---|---:|---:|---:|
| real尺寸 | 737 | 289/32 | 305/48 |
| real方向 | 781 | 145/39 | 155/49 |
| sim尺寸 | 572 | 58/0 | 58/0 |
| sim方向 | 572 | 6/0 | 6/0 |
| real_seq03尺寸 | 140 | 49/4 | 67/22 |
| real_seq04尺寸 | 597 | 240/28 | 241/29 |
| real_seq03方向 | 159 | 43/11 | 49/17 |
| real_seq04方向 | 622 | 102/28 | 105/31 |

当前real聚合尺寸/方向各少接受16/10坏，同时各少误拒16/10好；尺寸收益主要在seq03，seq04只差1坏，方向两视频差6/3坏。各组独立匹配数量，不能把逐视频score计数相加冒充域级score计数。相对冻结部署score_only的real尺寸349坏/10误拒、方向163坏/19误拒，simple错接较少、误拒较多，接受数也不同，不能仅比较错接数量。只有两条real TEST视频，且此前VAL优势未稳定，不能据本轮局部结果称跨视频稳定或统计显著收益。

**新旧前端与风险判断分开（事实）：** 同次正式B框的raw尺寸错误real249→midpoint379、sim26→58；raw方向错误real176→184、sim52→6。几何正确性变化在simple筛选前已经发生，可靠性从未写回最终框。逐帧real尺寸176个B好→midpoint坏、46坏→好，sim46好→坏、14坏→好；real方向88好→坏、80坏→好，sim1好→坏、47坏→好。real尺寸净增130坏，其中seq04为154→281（+127），seq03为95→98（+3）。这是固定TEST描述，不用于按视频设计方法或更换前端；不证明唯一几何根因。

旧B＋旧simple到midpoint＋新simple：real尺寸接受796→737、错误接受214→289、正确误拒28→32、状态准确率71.8277%→62.6310%；real方向接受742→781、错误接受132→145、正确误拒73→39、状态准确率76.1350%→78.5797%。sim尺寸错误接受26→58、方向52→6，全部接受不变。迁移同时改变几何与拟合风险参数，不能归因于simple模型单一因素。几何窗口第74节已经从TRAIN/VAL及正式几何证据记录了尺寸代价；RIoU/定位/角度改善不等于两条边都通过严格10%标准，本次计数与既有尺寸误差退化并不矛盾。

完整OBB：当前raw共同正确922/1440=64.0278%，simple完整接受1307/1440=90.7639%、共同正确899/1440=62.4306%；同次旧B raw共同正确1002、旧simple961。raw到simple损失23个共同正确观测，同时排除101个共同错误输出，体现筛选取舍，不是筛选把框改坏。新旧simple完整共同正确961→899，其中real466→389、sim495→510；不能以sim方向收益代表整条流程全面改善。

**基线定位与下一步（第19节用户决定已明确）：** 固定流程及TEST报告已补齐，本版B24/midpoint23/simple已确定为后续比较基线。基线不需要先优于score或达到理想准确率，已有错误接受、误拒和局部收益均为后续对照数值；它们不妨碍冻结基线，也不能写成新可靠性优化已经成功。继续冻结本版身份，不依据此TEST撤回B、挑域/视频策略、搜门限或重选权重；无需再重复当前固定TEST，也不继续相同三特征重拟合或重启高维structure共享分支。

可靠性开发下一项沿用第16节已有TRAIN/VAL建议，只改**图像尺寸参考的核心范围/长短边监督**：先固定可微监督实现、系数、同预算/同初始化对照和TRAIN角色，核验参考自己的几何精度、原有长短边双侧探针及真实错误辨识；通过有限TRAIN检查后，一次固定全量VAL评价参考精度、分域/逐视频同数score/simple对照与正确观测覆盖。保持midpoint、参考模板读取、坐标链和评价规则不变，所有新设计/选择仅由TRAIN/VAL决定；本次TEST只是冻结报告的外部现象，不用于制定修改。当前模板real VAL AUROC .360388、参考尺寸仅72/373两边10%内，且simple real尺寸VAL .591283，说明此前即缺乏充分参考精度/可分性证据；不是看到本轮TEST才提出该路线。参考几何更准能否转化为可靠性收益仍待验证，不能只要求heatmap loss下降、增加轮数或调整风险映射。

几何优化窗口若后续按其TRAIN/VAL规则保留新的前端，应按新身份迁移整个可靠性协议与拟合参数，再冻结后报告，不自动继承本版分数。当前不修改另一窗口或提前采用未完成参数实验。大小论文暂不修改；三个标志是观测使用建议，不是物理安全判定。**TEST已经多次暴露**，本轮不称未接触独立确认；接口保持不等于深度精度保证。

## 19. 用户正式固定比较基线与几何／可靠性改进分工（2026-10-04）

**用户决定（已执行）：** 当前B epoch24＋正式midpoint epoch23＋现有simple三标志及第18节固定TEST成绩作为后续可靠性比较基线。这里的“基线”表示已有、身份清楚、规则和结果可复现的比较起点，不要求最优或先达到高准确率。基线收尾已完成，不再把性能优化当成收尾前置条件。原权重/policy/门限、数据划分、评分定义和产物保持原字节；本轮仅更新四份现有说明文档，不修改算法、训练代码、父来源清单、原结果或实际论文稿。

**midpoint影响的准确表述（事实与推断分开）：** 当前固定TEST的尺寸几何达标率确实比同次B降低：real错误249→379、sim26→58；这些标签变化在可靠性判断之前已经发生。simple输入的最终长短边/长宽比随之改变，风险参数也已在TRAIN/VAL重新拟合，原B risk不能直接代表新前端。迁移后的尺寸状态准确率降低是整条“最终框＋新风险参数＋冻结门限”的结果；现有迁移对照不能把下降全部归因于判别器本身变差，更不能概括midpoint把三个分量都恶化：sim方向错误52→6，real方向状态准确率76.1350%→78.5797%。不能通过修改可靠性分支消除原框自身的尺寸误差。

以real尺寸为例，旧simple检出35/249坏＝14.0562%，新simple检出90/379＝23.7467%；错误总量增加130，检出的坏框增加55，故错误接受仍增加75（214→289）。这说明绝对错误接受与错误检出比例需要同时看；本次局部比例提高并未抵消几何代价，也不单独证明判别器的跨视频能力提高或下降。

**后续改进分工（建议，未实施）：**

1. 本可靠性窗口固定B24/midpoint23及基线policy，下一候选只改尺寸质量证据或判断。优先沿第16/18节已有TRAIN/VAL证据补强图像参考的核心范围/长短边监督；在线只预测“建议使用/拒绝”，不写回最终框。对照保持前端、图像、输出数量/score、坐标链相同，分域/逐视频检查错误接受、正确误拒、错误检出及正确观测覆盖。新的工作点若需设计，应在新版本中预先限定TRAIN/VAL目标及误判代价；原95%基线policy仍原样保留。
2. 要修复midpoint原框自身尺寸精度，属于独立几何窗口的TRAIN/VAL实验：检查四点几何读出、最终长短边监督及尺寸—中心—方向取舍。仅收紧可靠性门限会拒绝更多框，不能让尺寸变准；统一放大或按TEST视频回退会改变模型，不能作为本可靠性单因素优化。当前不在本窗口新增几何项、修改头或替换正在另行研究的参数版本。
3. 即使另一窗口将来依据TRAIN/VAL保留新几何前端，本版比较基线及其第18节结果仍保留。新前端另立完整身份并迁移风险参数，前端变化和可靠性变化分别对照，不能覆盖本基线或自动继承其成绩。本轮不因已暴露TEST挑选任何前端、结构或阈值。

当前已有可分性检查、参考训练、模板读取及全量VAL验证，不重复宣称这些仍未完成，也不重复同权重诊断来代替方法改进。下一版本的监督公式、系数、角色划分、有限TRAIN预算与继续条件仍须在新结果前确定；本次只固定基线和研究分工，尚未实施新训练。中心保留、等比例变换、原图坐标还原及深度接口约束沿用原协议；中心正确性或深度精度不因固定基线而被额外认证。TEST多次暴露状态继续披露。

## 20. 更换窗口接续摘要：只提高可靠性判断准确性（2026-10-04）

### 20.1 本窗口收束与新窗口目标

**用户决定：** 当前小论文流程已经闭合；B epoch24＋正式midpoint epoch23＋现有simple三个独立标志及第18节固定TEST结果，正式作为后续可靠性优化的比较基线。基线是可复现的比较起点，不要求先表现最好，也不因存在误判而反复收尾。新窗口固定研究目标为**提高中心、尺寸、方向使用判断的准确性**，优先推进尺寸这一项范围有限的改进，随后再决定是否推进方向；不在新窗口更换检测前端或修复原框几何。

本窗口早期还涉及D、E/E-H、F-S等检测几何工作，其实现、结果及保留结论继续由[独立几何交接](geometry_precision_handoff_20261001.md)承接。可靠性已按用户要求剥离到本文；新窗口不能将两个任务重新合并，或自动采用几何窗口后续未完成的版本。连续状态接口、物理摆角/深度/安全报警属于大论文后续内容，实际大小论文稿本阶段暂不修改。

**本次交接操作：** 仅补本文的接续摘要和顶部阅读入口，保留原章节、旧身份及归档链接；不修改算法、权重、policy、门限、来源清单、结果文件或其他文档。不新增训练、推理、TEST、论文搜索或本地结果副本，不连接服务器。上一轮已修改的README、OBB说明及流程记录保持原样；未提交状态不是实验尚未完成的证明。

### 20.2 本窗口可靠性工作已经完成到哪里

| 阶段 | 已完成事实 | 对后续的含义及证据位置 |
|---|---|---|
| 标注与真实TRAIN支持 | 核对2558帧TRAIN；补全real原生结构文件后1810帧转换一致，sim748帧使用Webots OBB；固定训练方向资格并收集原B真实输出 | sim没有原生轴线不等于没有方向GT；原B真实坏中心real/sim为1/0、坏尺寸54/2，错误支持稀少。旧记录见归档R1.1—8 |
| geometry/普通ROI/structure正式分支 | 冻结原B，三臂同预算8轮、各20464步，固定epoch08；完成保存重载、VAL缓存及online TEST入口与复核 | 普通ROI有局部结果，但未稳定优于score；structure没有成立的整体增量，均未进入当前simple基线。见归档R1.9—12、20—22 |
| 结构机制与候选辨识 | 固定8个TRAIN视图完成质量/结构梯度检查、112个候选短拟合、方向显式关系检查 | 局部共享梯度冲突和参考偏差存在，但不是唯一根因；loss下降没有证明双侧辨识。方向最终零控制/真实关系/理想关系分别6/16、6/16、7/16正确配对，双侧正确0/8、0/8、1/8。见归档R1.12—17 |
| 回到简单三标志 | 按用户澄清恢复简洁接口；新建B适配的simple，不把ordinary ROI改名为旧Base V3/V5.1；完成TRAIN拟合、VAL工作点、smoke及固定TEST | 中心保留有效输出；尺寸/方向独立风险与拒绝。旧B TEST是历史对照，见本文第1—9节及归档R1.23—30 |
| simple可分性 | 复用TRAIN/VAL缓存，核对排序、混淆、同数量score、跨视频配对贡献和错误支持差异 | 原B real VAL尺寸/方向simple AUROC .589500/.669308低于score .686285/.725558；pooled指标不能代替同视频辨识。见第8/9节 |
| PQA启发的独立尺寸参考 | 完成理想读取、真实smoke、384张×4轮参考学习、固定epoch04及432张参考留出/全VAL评估 | 参考是OBB派生二维几何代理，不是可见轮廓GT；独立参考学习与显式一致性没有等同复现PQA或已验证新贡献。原二阶矩尺寸读取未达到需要的精度。见第10/11节 |
| 迁移正式midpoint前端 | 完成check/probe/collect/fit/smoke、来源及数值复核；以完整正式VAL选定的midpoint23为前端，重新拟合simple | 前端改变几何、特征和真实错误分布，旧B policy不能直接代表新前端。见第12—14节 |
| 固定模板读取与全量验证 | 固定半峰核心、log-Gaussian IRLS读取，在432参考留出＋887 VAL共1319帧验证，原模型/框/policy不变 | 模板比旧矩更接近GT，但real错误辨识仍未成立，没有替换在线simple标志。见第15/16节 |
| 当前基线TEST闭合 | 完成CPU缓存入口及1440帧固定报告，逐帧、来源、汇总、配对B结果复核通过；用户明确固定此版为比较基线 | 工程成功和判别准确性提高分开；本版收尾已完成，不再重复当前TEST。见第17—19节 |

表中的归档统一指[替换前完整可靠性交接](archive/20261003_reliability_replaced_by_simple_v1/readable/docs/reliability_handoff_20261003.md)。原Base V3/V5.1含旧检测/DINO/历史输入，与当前simple不是同一模型；其历史结果不得改名继承。旧ROI、structure及方向短拟合的局部收益和负结果均保留，不概括为“结构方法永远无效”，也不重启原v1路线作为默认下一步。

### 20.3 新窗口必须冻结的身份与接口

| 项目 | 服务器项目根目录下的相对位置 | 绑定身份 |
|---|---|---|
| B配置 | `crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py` | SHA `9da972b7010540e12b2501d1c400db13c159381f495f539d6315e7adb498e345` |
| B权重 | `work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth` | SHA `8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23` |
| 正式midpoint | `work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/head_epoch_23.pth` | SHA `2c4c5ae9e071cbdb13fb722f4b2d3986861c62bd60c53568c72fb9b02d662f0b` |
| 正式选权证明 | 同目录`selection.json` | SHA `5e4b24eb15f8fcc925c95445d1f6ad3c7fa643b350267b3647710a0d14024ec6`；原完整VAL选权，不使用TEST |
| 当前simple | `work_dirs/port_midpoint_reliability_v1_fit/policy.json` | SHA `1adac499369bade5ef1f79495260002697d2b698c80ebf5424d3d77438fae4af` |
| 已研究的尺寸参考 | `work_dirs/port_size_reference_v1_train/epoch_04.pth` | SHA `72b621fe2f0b78ac5292f2142eb318ab256791ab87ce54eee75e50f217e0673a`；候选身份，不是当前基线组成 |

当前simple只看预测score、相对尺寸和长宽比，两个独立线性sigmoid风险模型共8系数；TRAIN拟合，VAL原95% pooled规则产生工作点，尺寸≤.39773387607871297、方向≤.4143115365368443，固定score-only风险`1-score`≤.7043236196041107。这些风险不是校准错误概率。中心标志保留每个有效最终输出，**尚未训练独立中心正确性预测器**；不能称三个分量均已经具备有效错误辨识。

在线输出仍是原图最终OBB/score＋`center_accepted/size_accepted/angle_accepted`。无输出保留该帧并给三个false；拒绝尺寸/方向不删除中心，不更改框、score、检测输出数量。GT仅用于TRAIN监督及离线评价，绝不进入在线判别；域、视频ID、历史或未来帧也不进入当前API。新候选另立版本，保留本版policy、工作点、源码来源及原结果字节，不能覆盖基线来制造改善。

### 20.4 决定下一步所需的关键证据

**事实——当前TEST比较基线，完整表见第18节：** real输出859/868、中心正确858/859、全帧中心正确858/868；sim三项均100%。中心有1个错误输出仍被接受，9帧缺测未修复。下表状态准确率仅统计输出帧，状态误判＝错误接受＋正确误拒。

| 域/分量 | simple接受/总帧 | 错误接受 | 正确误拒 | 状态准确率 | 同数score错误接受 |
|---|---:|---:|---:|---:|---:|
| real尺寸 | 737/868 | 289 | 32 | 62.6310% | 305 |
| real方向 | 781/868 | 145 | 39 | 78.5797% | 155 |
| sim尺寸 | 572/572 | 58 | 0 | 89.8601% | 58 |
| sim方向 | 572/572 | 6 | 0 | 98.9510% | 6 |

real同数score局部比较有收益，sim全部接受没有检错收益；不称跨视频稳定或显著。当前raw完整共同正确922/1440，simple完整接受1307/1440、共同正确899/1440。拒绝让可用性改变，不会修复框。相较同次B，midpoint原框尺寸错误real249→379、sim26→58，方向real176→184、sim52→6；尺寸代价先于可靠性判断发生，但不能把三分量都称为恶化，或把迁移后下降全部归因于simple模型。

**事实——下一设计依据必须来自既有TRAIN/VAL：** 新midpoint TRAIN尺寸46坏/2512好、方向153坏/2405好，坏样本均来自real；VAL出现real/sim尺寸188/56坏及方向61/34坏。simple real VAL尺寸/方向AUROC .593428/.635992，score .710278/.711203；sim simple .700149/.652781、score .630013/.618385。训练内低错误率与VAL辨识不足已有证据，不能单靠增加模型大小或重训同一三特征预期解决。

**事实——图像尺寸参考尚不够准确：** 固定模板使参考最大长短边平均相对误差由旧矩real VAL131.1518%降至19.1554%、sim121.6411%降至17.3256%，但两边均10%内仅72/373、197/512。real共同可用373输出上的模板AUROC .360388，低于simple .591283及score .708786；sim模板 .682253，有局部收益。参考几何更准与可靠性排序更好不是同一结论，现模板不得作为已成功的替代标志。参考留出432帧的midpoint真实尺寸坏框real4、sim0，不能据少数错误/单类数据证明泛化。

**机制推断，尚非唯一根因：** 三特征缺少直接图像范围比较，真实TRAIN坏框稀少且误差分布转移，可能限制判别。固定14图框外响应占比中位21.8%却贡献短轴矩中位77.5%，读取替换明显减少参考尺寸偏大，支持尾部污染是旧矩的一项重要因素；模板仍不准，说明仅改读取未解决参考范围学习。参考与错误框一致也可能让错误获得低风险。现有证据不足以将容量、类别不平衡、域代理、某个损失或标注因素认定为唯一根因。

### 20.5 下一项建议及仍未实施的部分

**最推荐的下一步：** 在冻结B24/midpoint23和现有对照的条件下，设计一项**独立图像尺寸参考的核心范围/长短边监督改进**。沿用当前图像的detached P3侧分支，参考只作为尺寸判断证据，不写回框；先改善参考自身精度，再检验真实好/坏尺寸是否更可分。固定模板读取作控制，避免一次同时更换图像分支、读取器、风险模型及门限。此前文献借鉴与边界见第10节及归档R1.26：PQA提供“图像几何参考与当前框一致性”的思路，IoU-Net等说明分类score与定位质量应区分；这些论文没有验证本项目三分量适配或新监督的收益。

新窗口先复用已有代码、TRAIN角色和VAL结果形成一项可审查设计。**第20节交接时，新增监督的公式、系数、是否可微、预算和量化继续条件尚未固定；第21节已固定设计，第21.8节已按用户补充实现有限预检和两臂入口。随后真实GPU两次在梯度核对处失败，历史修复见21.9，当前gradfix2及失败帧预检见21.10。** 不直接把非可微IRLS模板当作可反传尺寸loss；若新增可微读出，须说明它与固定评价读出的职责以及为何能监督范围。原参考的384拟合/432参考留出、guard及未见标签边界已经明确，可优先复用；B/simple已见整个检测TRAIN，不能称全前端独立留出。任何角色或预算改变先记录，不依据真实错误标签选择有利样本。

设计固定后再按用户授权本地实现、复核、提供服务器指令；顺序为有限TRAIN数值/梯度/原输出隔离和保存重载检查→同初始化/同预算参考监督对照→固定全量VAL。需要检查参考本身的长短边误差、扩大/缩小双侧响应、真实错误排序、同接受数错误接受/正确误拒、正确观测覆盖与不可用支持；以分域/逐视频结果为主，不能仅靠heatmap loss、pooled AUROC或接受数降低宣称成功。实际精度/检错/覆盖的数值门槛须在新结果前确定，本文没有替新版本先宣布通过。

中心高覆盖与当前标志保持；真实坏中心支持太少，暂不强行训练中心门控。方向结构v1不作为默认重启项；尺寸路线有效后再另定方向证据和目标。三分量仍独立：一项尺寸改进不能冒称整个可靠性全面提升。

### 20.6 新窗口最少阅读与服务器产物定位

按需阅读下列代码及其同名前缀`_protocol.json`、`_sources.json`、对应测试；不遍历整个项目重做审计。新增核心曲率入口见21.8，历史修复见21.9，gradfix2及原运行顺序见21.10；已完成服务器回传、R/Q1判断和下一步建议见21.11，不重复启动本次训练。旧基线入口保持原身份。

| 职责 | 当前关键代码 |
|---|---|
| 冻结TEST基线与来源约束 | [eval_port_midpoint_simple_reliability_v1_test.py](../crane_project/tools/eval_port_midpoint_simple_reliability_v1_test.py)；[对应测试](../tests/test_port_midpoint_simple_reliability_test_v1.py) |
| midpoint迁移及模板读出 | [run_port_midpoint_reliability_v1.py](../crane_project/tools/run_port_midpoint_reliability_v1.py)、[port_midpoint_reliability_v1.py](../crane_project/utils/port_midpoint_reliability_v1.py) |
| simple模型及三个标志 | [port_simple_component_reliability_v1.py](../crane_project/utils/port_simple_component_reliability_v1.py) |
| 独立尺寸参考监督/坐标/阶段 | [run_port_size_reference_v1.py](../crane_project/tools/run_port_size_reference_v1.py)、[port_size_reference_v1.py](../crane_project/utils/port_size_reference_v1.py)、[port_size_reference_v1_torch.py](../crane_project/utils/port_size_reference_v1_torch.py) |
| 固定模板全量评价 | [eval_port_midpoint_size_template_v1.py](../crane_project/tools/eval_port_midpoint_size_template_v1.py)、[port_midpoint_size_template_v1.py](../crane_project/utils/port_midpoint_size_template_v1.py) |

服务器项目根目录为`/media/omnisky/personal_files/ljj/symEOOD`。已有主要目录：

- 正式几何ROI缓存：`work_dirs/port_geometry_midpoint_formal_v1_roi_cache`，当前迁移使用`train_s1.pt/val_s1.pt`；不用半尺度结果冒充正式选择。
- midpoint迁移：`work_dirs/port_midpoint_reliability_v1_{check,collect,fit,smoke}`，修复后的probe为`work_dirs/port_midpoint_reliability_v1_probe_fix_v1`；原fit/collect先于probe接口修复，已审旧合同的精确pin见第15节，不能重写旧来源清单放行。
- 参考训练与读取：`work_dirs/port_size_reference_v1_train`及原assess目录；模板全量结果`work_dirs/port_midpoint_size_template_v1_assess/assessment.json`。
- 当前固定TEST：`work_dirs/port_midpoint_simple_reliability_v1_test_check`、`work_dirs/port_midpoint_simple_reliability_v1_test_cached_v1`；正式框来源`work_dirs/port_geometry_midpoint_formal_v1_test_eval`，图像尺寸元数据来源`work_dirs/port_reliability_branches_v1_test_cached_v1`，不使用后者旧ROI质量替换simple。

上述位置由已有入口/回传记录定位；新窗口不连接服务器检查文件是否仍在。缺少字节复核材料时明确限制，不能从文档SHA声称已经重新加载GPU权重。最近三个回传包为`port_midpoint_reliability_v1_review_20261004_150523.tar.gz`、`port_midpoint_size_template_v1_review_20261004_161835.tar.gz`和`port_midpoint_simple_reliability_v1_test_review_20261004_190008.tar.gz`，已有身份/核验见第14/16/18节；仅在确有新核对需要时从用户附件内存读取，不再建本地解包/失败复核目录。

### 20.7 后续不可改变的证据边界

- 所有新设计、权重/门限/候选选择依据TRAIN/VAL；**当前TEST已多次暴露**，不能称未接触的独立最终确认。之后可对预先冻结新版作描述性TEST报告，但不以当前TEST调参、按视频回退、换权重或选择分支。
- 中心正确率仅统计输出帧，另报原输出覆盖和全帧中心正确覆盖；分量误判、接受后正确率、错误检出、正确保留、接受覆盖及完整OBB共同正确分别报告。AUROC/AP是错误排序指标，不是状态准确率。
- score-only冻结部署门限与同接受数score排序对照分开；同分边界、共同参考可用集合、全输出/全帧分母及不可用错误支持不可省略。GT评估资格不进入在线接受判断。
- 保持等比例变换、原图坐标链、π周期/宽高交换等价及深度接口约束；接口未改不意味着深度或物理安全精度得到保证。
- 不重新打开DINO、蒸馏、检测候选排序、原高维structure拼接/方向短拟合、完整审计和大论文连续状态路线。已有代码逻辑错误可修，但不能以反复诊断替代新的范围明确改进。
- 本地按授权改代码并完成必要复核，训练和GPU检查由用户在服务器运行；输出用新目录，保留原结果与源身份。不主动打包提交源码、不持久保存新的回传结果副本。可靠性记录集中更新本文，不覆盖独立几何交接或实际大小论文稿。

### 20.8 可直接粘贴到新窗口的接续指令

```text
请接续 /Users/mac/Documents/paper/symEOOD 的可靠性准确性优化。
先读取 docs/reliability_handoff_20261003.md 第20节，再按需读取第1/14/16/18/19节和第20.6节关键代码，恢复上下文，不重复完整审计。

已固定比较基线：B epoch24＋正式midpoint head_epoch_23＋work_dirs/port_midpoint_reliability_v1_fit/policy.json 的现有simple三标志。固定TEST流程已经完成，完整结果见第18节；基线不要求先优于score。原权重、policy、门限及结果保持，不能覆盖。
本窗口唯一目标：提高中心/尺寸/方向使用判断的准确性，优先做尺寸一项有限改进，不更换前端、不写回检测框，不重开几何优化。实际大小论文和连续状态接口暂不修改。

已完成simple可分性、独立参考训练、模板读取及全量VAL验证。模板改善自身尺寸范围，但real可靠性辨识未成立；现有TRAIN/VAL已支持继续研究参考核心范围/长短边监督，不能认定唯一根因。旧ROI/structure及方向短拟合暂不重启，也不只延长训练或重拟合同一三特征。
请先基于已有证据给出一项机制明确、可微实现可审查的下一版本设计，固定监督公式/系数、同初始化同预算对照、TRAIN职责和量化继续条件，区分事实、推断和待验证；本次先不启动训练。若推荐改变既有路线，请说明已有证据支持及范围。

新设计与选择仅用TRAIN/VAL，披露TEST已多次暴露，不用TEST调参或重选模型；分域/逐视频比较错误接受、正确误拒、同接受数score/simple及正确观测覆盖。中心命中仅统计输出帧，另报输出覆盖和全帧中心正确覆盖。GT仅用于监督/离线评价，sim Webots OBB可监督方向，不要求轴线标注。
保留等比例变换、原图还原、宽高交换/π周期和深度接口约束。不要连接服务器；后续经授权在本地改代码、必要测试与审查后给服务器指令。记录集中维护本交接，不新建本地结果副本，不修改几何窗口或实际论文稿。
```

## 21. 尺寸参考核心曲率监督：设计与有限预检入口（2026-10-04）

### 21.1 本轮恢复与必要静态核对

**已完成事实：** 先读第20节，再定向读取第1/10/14/16/18/19节、20.6列出的相关实现/协议/测试和归档R1.26的文献借鉴边界；没有重新遍历项目或完整审计。四份相关源码清单的直接`source`文件共18项SHA核对全部相符（尺寸参考5项、midpoint迁移5项、模板全量评价4项、固定TEST入口4项）；这是当前本地直接文件核对，不是再次证明全部父合同、服务器缓存或权重字节。对应服务器policy、参考04和midpoint23权重本地不存在，本轮没有重新加载它们，也没有连接服务器。

当前`SizeReference`为detached P3的16通道侧分支，原训练只有`reference_loss`像素热图项；网格仍为`(j+.5)*2−4`，目标sigma=L/4、S/4。`template_reference`在预测半峰连通核心上拟合`log(response)`的二次曲面，经负定曲率恢复长短边；它包含背景中位数、硬核心/连通选择、IRLS及不可用保护，不能直接作为可微loss。现有评价代码已经分开共同可用集合、参考自身可用集合和全输出集合，也已经把不可用当拒绝的代价列出。原尺寸参考runner的`assess`仍评价旧B上下文和矩读取，不能直接用于本新候选的正式midpoint比较；后续新入口须复用已核验midpoint配对/模板评价链，而不是改写旧入口的来源身份。原中心/方向标志、坐标还原和最终框保持约束沿用，不修改这些入口。

**正式midpoint的收益继续保留：** 正式midpoint已在固定TEST改善定位、sim角度和两域RIoU；部分尺寸及时序指标存在代价。后续size/motion辅助项没有一致收益，不能据此否定正式midpoint。该几何事实来自已核验交接，本轮不重做几何审计、不采用几何窗口后续候选。可靠性筛选只能改变建议使用状态，不修复框。

本轮只补本文的设计记录，保留所有已有修改。未改代码、源码清单、协议JSON、权重、policy、门限、原结果、几何交接或大小论文；未启动训练/新推理/TEST，不新增本地回传结果副本。以下数字为**预先固定的设计门槛，不是新实验成绩**。

### 21.2 已知问题、机制假设与范围

**设计依据只取TRAIN/VAL：** 第16节固定模板在real VAL的参考误差仍平均19.1554%，仅72/373两边均10%内；真实坏框AUROC .360388，低于simple .591283和score .708786。好检测的参考短边偏大、坏检测与参考同时偏小是已观察到的分组特征，尚未证明逐帧相关性或唯一根因。第14节固定14图的尾部响应对短轴矩贡献过高，且更换读取降低参考误差，支持旧矩尾部偏差；但模板仍不准，不能继续把仅换读出当作充分解决。新midpoint TRAIN只有46个尺寸坏框且sim没有坏框，直接扩充质量分类器缺少充分错误支持。

**唯一候选设计：** 暂名`size_core_curvature_v2`，只在原参考学习中补一个“GT核心内的归一化log曲率监督”。GT长短边通过Gaussian曲率进入监督，让过宽和过窄的参考都有明确尺度梯度；所有拟合图使用OBB监督，不按检测错误标签挑图或重复坏框。不增网络容量、不增加另一直接回归头、不加质量MLP、不回归检测框、不换风险聚合。原像素项继续约束响应位置、幅值、外围及背景；新项针对固定模板所依赖的核心曲面形状。**待验证假设：** 核心曲率更准确，可能减少参考—错误框一起偏小及好框参考偏大的情形，从而改善显式尺寸一致性的判断。它不保证解耦冻结B特征与检测误差，不宣称已经找到唯一根因。

复用已核对文献的有限启发：[PQA作者原文](https://arxiv.org/html/2511.08186v1)采用OBB派生Gaussian位置热图及像素一致性衡量整体定位质量；[IoU-Net官方入口](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.php)区分分类置信与定位置信。本轮定向重新打开这两份原文，没有扩展文献检索。这里的核心曲率loss、独立尺寸标志和冻结侧分支都是本项目拟议适配；不是PQA原损失/整体min-max聚合复现，也不继承两篇论文的性能或新颖性结论，不引入NMS、候选重排或框优化。

### 21.3 监督公式与可微实现

在既有stride2有效网格上，先按现有等比例/反射链把GT OBB映射到模型坐标，规范长短边为L≥S，沿GT长短轴的中心相对坐标为u、v。设

\[
\xi=4u/L,\quad\eta=4v/S,\quad
t=\exp[-(\xi^2+\eta^2)/2]\,\mathbf 1_{|u|\le L/2,\ |v|\le S/2},
\qquad C=\{\text{valid}\land t\ge0.5\}.
\]

这与原目标和半峰含义一致，C由**TRAIN GT**确定而不依赖预测概率，不对预测阈值或连通操作求导。它是标注派生的Gaussian核心，不是可见轮廓GT。sim直接使用Webots OBB，不要求补轴线。C上令`z`为现有分支logits，`y=logsigmoid(z)`，令每行设计向量为`X_i=[1,ξ_i,η_i,ξ_i²,ξ_iη_i,η_i²]`，固定权重`w_i=t_i/Σ_C t`。拟合

\[
\beta=(X^TWX)^{-1}X^TWy,
\qquad
K=\begin{bmatrix}-2\beta_3&-\beta_4\\-\beta_4&-2\beta_5\end{bmatrix},
\qquad K^*=I_2.
\]

β使用从0开始的下标。理想`log t`的系数为`[0,0,0,−1/2,0,−1/2]`。若参考是同轴Gaussian、长短边为L_ref、S_ref，则`K_11=(L/L_ref)²`、`K_22=(S/S_ref)²`；因此把两项推向1分别监督长边和短边范围，而非只监督面积或几何平均尺寸。一般情况下物理精度矩阵为`D^{-1}KD^{-1}`，其中`D=diag(L/4,S/4)`；仅当K正定时才有该Gaussian的有限尺寸解释。**训练不对K做求逆、特征值开方或负曲率截断**，从平坦/非Gaussian响应开始仍可得到纠正曲率的梯度。常数项及线性项自由拟合，新项不另加中心参考或方向使用标志的监督；K的交叉项只是保持原OBB Gaussian曲面形状的一部分。

定义固定SmoothL1函数`h_δ(a)=a²/(2δ)`（`|a|≤δ`），否则`|a|−δ/2`，δ=0.1。唯一新增监督模块由曲率与拟合残差两项组成：

\[
\mathcal L_{curv}=\tfrac12\{h_{0.1}(K_{11}-1)+h_{0.1}(K_{22}-1)\}+h_{0.1}(K_{12}),
\quad
\mathcal L_{quad}=\sum_{i\in C}w_i h_{0.1}(y_i-X_i\beta),
\]
\[
\boxed{\mathcal L_{new}=\mathcal L_{v1}+0.25\mathcal L_{curv}+0.05\mathcal L_{quad}}.
\]

`L_v1`完全沿用现有soft focal BCE（正0.25×|t−p|、负0.75×p²、有效目标质量归一化）；不更改目标或其归一化。`L_quad`避免仅平均曲率满足监督、核心却不是合适二次曲面的情况。0.25/0.05、δ=0.1都是本次设计先验，在新结果前固定，并非已测得最优；本版不搜索系数、不延长预算、无warmup或动态loss平衡。

**拟议可微实现：** GT几何/掩码/权重均为常量；用float64 QR或等价稳定线性求解预计算固定投影`P=(X^TWX)^{-1}X^TW`，不在代码中显式求逆。随后Torch执行`y=F.logsigmoid(logits_C).double()`、`beta=P@y`和上述loss；从y到logits、stem及输出卷积保留计算图，不转NumPy、不detach。只有GT投影可离线NumPy计算。使用`logsigmoid`避免`log(clamp(sigmoid))`在低概率处梯度截断；不对β/K作人为可信裁剪。GT核心不足16格、GT模型短边<16px、加权X秩不足6或`cond(sqrt(W)X)>1e6`时，该图只保留原像素loss并记录原因，不删除图、不另补图；所有跳过条件仅由TRAIN GT几何决定，不能按预测好坏跳过。任一域全无合格核心则停止该设计的训练放行。记录各域/视频合格与跳过数量。

**训练代理与固定读出职责不同：** 新loss在GT核心拟合`log p`；部署/评价仍在预测半峰连通核心拟合`log(max(p−median_background,0))`，使用原IRLS。两者共享曲率—尺寸关系，却不完全相同。核心位置偏移、背景扣除、非Gaussian形状、分辨率和跨视频变化仍可能让训练代理改善而真实读出失效，必须用固定模板输出验收；不声称该loss直接优化了非可微IRLS或真实错误AUROC。不将训练β或GT核心送入在线接口。

### 21.4 同初始化、同预算对照及数据职责

| 项目 | 预先固定设计 |
|---|---|
| A0配对控制 | 原`SizeReference`＋原`L_v1`；新目录重新做公平配对，不覆盖旧参考04 |
| A1唯一候选 | 相同网络＋`L_v1+0.25L_curv+0.05L_quad`；只改监督模块 |
| 初始化 | seed1701；生成一个原架构初始state_dict，两臂逐字节复制并验证相同SHA；优化器各自从空状态开始；不从旧epoch04续训 |
| 拟合数据 | 沿用原384张：real_seq01/05/06/12各64、sim_seq08拟合段128；按序列/帧序取样，读取同一已固定列表和变换 |
| 预算 | 每臂4轮×384=1536步，batch1；两臂合计3072正式更新；Adam lr0.001、weight_decay0、clip10；相同epoch内图序`RandomState(1701+epoch)` |
| 预处理与冻结 | 原干净单尺度等比例预处理，无新增增强；B24/正式midpoint23与BN冻结，参考输入只取detached原P3；不让新loss进入前端 |
| 选权 | 两臂固定epoch04，无最佳epoch选择；每轮保存/重载校验；smoke各4步且丢弃，不计为正式模型 |
| 参考留出 | 原432张：real_seq13整段304、sim_seq08尾部128；保留32张guard；剩余TRAIN不拟合，不增加或按坏框标签换图 |
| 留出职责 | 两臂及系数/预算/读出锁定后，仅评价参考误差、可用性及机制探针；不拟合权重、loss、风险或门限，不据留出改选图/选epoch |
| VAL职责 | 固定全部887张，进行一次两臂全量配对评价及预定义全局覆盖门限；不改loss/系数/参考读取或挑视频回退 |

旧epoch04＋固定模板、第14节冻结simple及score保留原身份；旧epoch04是历史参照，A0是新配对监督控制，不能把二者混为一次相同初始化实验。两臂的新初始状态、样本/顺序、输入、步骤、损失分项、梯度和保存重载写入各自新版本服务器产物；初始化相同不等于必然与旧epoch04完全数值重现。需报告参考训练增加的时间、显存和固定推理延迟，不能把同更新数说成计算成本完全相同。单seed、少数视频只支持有限可行性，不声称统计显著或独立泛化。

B/simple已经见整个检测TRAIN，参考留出只对参考学习器成立；sim留出与拟合仍是同视频的分隔片段。参考留出当前real只有4个midpoint尺寸坏框、sim为0坏，主要用于参考精度及探针检查，不凭它证明真实错误检出泛化。尺寸监督本身使用所有拟合图的OBB，不需要新增真实坏框、蒸馏或合成质量标签。

### 21.5 固定一致性、候选标志与评价分母

两臂都继续用当前最终midpoint框决定上下文，原模板参数原样保留：半峰0.5、至少16格、10步IRLS、Huber log δ=.15、最大log RMSE=.15、原边缘/弱证据/短边保护。新loss的SmoothL1 δ=.1与读出δ=.15职责不同，不互换。背景、核心、风险聚合及不可用条件不做参数搜索。

参考与当前框的显式尺寸一致性仍为

\[
d=\max\{|\log(L_{pred}/L_{ref})|,|\log(S_{pred}/S_{ref})|\},
\qquad r=1-\exp(-d/0.1).
\]

不反转real风险、不分域拟合、不加新simple特征或融合系数；长短边各自有符号log差另外报告以免max掩盖非主导边。尺寸在本有限实验中的比较决策为`template_defined && r≤τ`，只用于固定工作点的离线评价，不生成最终部署policy。中心仍接受每个有效原输出，方向完整沿用冻结simple；无输出三个false，参考不可用只使尺寸比较决策拒绝，独立报告代价，不回填GT或默认为可信。没有新中心门控、方向实验或深度输出。

**固定比较工作点的规则提前固定（用户补充，已采纳）：** 对每臂全部VAL的参考可用输出，以无GT标签的pooled 95%全帧覆盖规则取第`ceil(.95×887)=843`个最小风险为τ，边界并列全部接受。可用输出不足843则记录目标不可达，不放宽读出保护、不改覆盖目标或临时回退simple。τ只保存在比较报告中，不生成新部署policy、不写回原policy；不以VAL正确性优化τ、不按域/视频设门限。冻结基线门限继续是size .39773387607871297、angle .4143115365368443及score .7043236196041107。两臂比较阈值不同只允许来自相同预定义覆盖规则，不能来自错误标签。**95%不是最终判断准确性目标**：886输出中仅642尺寸正确，至少接受843个时FA下界为843−642=201；当前simple FA212，因此同一数量的理想排序最多改善11个。并列使接受数更多或参考不可用排除好框时，下界还可能提高。最终尺寸标志工作点须在排序收益成立后，按预先规定的正确保留/误拒约束在TRAIN/VAL另立版本，本轮不确定或部署该工作点。

评价各域/逐视频都报告`FA=坏而接受`、`FR=好而拒绝`、`ED=坏而拒绝`、`CR=好而接受`。输出帧状态准确率为`(ED+CR)/N_output`；错误检出率`ED/(ED+FA)`、正确保留率`CR/(CR+FR)`、接受后正确率`CR/(CR+FA)`；接受覆盖`(CR+FA)/N_all`、全帧正确尺寸覆盖`CR/N_all`。缺输出单列，不将漏检填进四格或当可靠性成功。AUROC/AP以坏框为正类，仅作排序证据。中心继续同时报`N_output/N_all`、`center_hit/N_output`和`center_hit/N_all`，不能只报条件命中率。

同数量对照包含score、冻结simple、A0和A1，分别做各组自己的接受数匹配；域级排序不由逐视频数字相加代替。报告共同可用、各臂自身可用和全输出三种支持，附各自/交集样本数及指纹、不可用好/坏数与原因；全输出比较把参考不可用当尺寸拒绝，score/simple可使用所有输出。若所需接受数不可达，报告实际数量并判该比较不可达，不能缩小到有利共同集合后仍称原覆盖通过。并列按risk、image名字固定无标签顺序，附错误接受最小/最大边界；胜负判定按对候选最不利的并列边界。

### 21.6 预先确定的继续条件与失败处理

以下是本有限版本的工程/机制/收益门槛，全部在任何新训练结果前确定；不用于选择多个loss/epoch。**阶段E**通过只允许完成两臂固定对照；**R**判断参考精度，**Q1**判断真实错误排序，**Q2**只描述95%固定工作点取舍，不作为最终部署工作点通过条件。R与Q1均通过才建议另立受正确保留/误拒约束的最终工作点版本；不能因Q2不合适自动否定已成立的排序收益，也不能因Q2通过就宣称最终标志已准确。冻结基线保留，不自动替换或启动TEST。

1. **E：实现与隔离检查（后续授权后做，目前未执行）。** 在原14个拟合角色视图中，对几何可评核心核对`log t`拟合K=I（float64绝对误差≤1e−6）；同轴±15%长/短边合成Gaussian的K符合平方尺度比，曲率梯度把偏大/偏小都推向GT；有限差分相对误差≤1e−4（非零梯度比较，避开SmoothL1分段点），真实4步smoke各分项有限且stem/输出梯度非零。覆盖padding排除、宽高交换/π等价、等比例/反射与原图还原。检测/正式midpoint参数与缓冲SHA不变、侧计算前后原框/score/输出数量精确一致、中心/方向标志逐帧不变；保存重载输出一致，GT不进入在线函数。只检验新增loss及新接口的受影响约束，不把原理想读取重跑当新收益。工程失败先修实现，不接触留出/VAL调loss。
2. **R：参考范围确有改善。** 以新A0为主配对控制，real VAL在A0/A1共同可用输出上的最大长短边相对误差均值至少相对降低20%，P90不升，两边均10%内比例至少提高10个百分点；全输出的“两边均10%内且参考可用”占比也至少提高10个百分点，防止只靠剔除难帧。sim VAL及参考留出两域的均值/P90不升、全输出参考正确覆盖不降；各域参考可用数不低于A0。逐视频全部报告，VAL每条视频的全输出参考正确覆盖不降。GT aspect≥1.2的共同可用输出上固定GT/长边±15%/短边±15%探针，各边单独绝对log差的双侧辨识比例real VAL及real参考留出均≥60%，且不低于A0；报告分母及不可用数量，不将探针当真实坏框。控制固定同图上下文，原中心/方向探针与等价表示仍不得改变尺寸一致性。边长跨越规范顺序的近方框不放进该双侧机制分母，但仍保留在所有真实尺寸评价中。
3. **Q1：同接受数的真实判断改善。** 主工作点预先固定为各组**冻结simple的接受数**，real VAL为333、sim为510，视频数量从原已审VAL决策记录读取而非由新成绩挑选。在全输出比较中，A1的real错误接受比score、simple和新A0三者中的最好者至少少5个（同数时正确误拒也必然至少少5个）；sim不劣于三者中的最好者。每条VAL视频同数时不劣于score/simple较好者，real至少一条严格改善；不可达不能算通过。在同一A0/A1共同可用支持上，A1真实real坏框AUROC另须至少高于score/simple中较高者0.02；它不能代替同数改善。90%/95%全帧名义数量继续按现协议列作固定次级诊断，不挑其中最有利一点作为主结论，不强求所有次级点都显著。参考留出坏框太少/无坏，不设AUC检错通过结论。
4. **Q2：95%固定比较工作点的正确观测取舍。** 用21.5规定的一次全局τ，报告全部VAL每域、每视频的CR/FR相对冻结simple变化、real FA是否至少减少5个及sim FA是否增加，并检查各域接受覆盖下降是否超过1个百分点、pooled 95%是否可达。所有原框、score、输出覆盖、中心正确覆盖及方向标志必须保持。另按候选每组实际接受数量匹配score/simple/A0，完整报告FA/FR/ED/CR及并列界。它只描述该覆盖约束下的取舍；不生成最终尺寸标志或以它单独阻断下一工作点研究。共同可用子集改善与全输出代价必须分别报告。

计数门槛按整数精确比较；浮点“不升/不降”只容许1e−10数值容差，不作为统计误差容忍。20%、10个百分点、5帧、0.02、60%和1个百分点都是本次有限可行性设计的实用门槛，不是显著性或安全标准。特别是sim冻结simple已经保留全部正确尺寸，主高覆盖点无法靠增加正确保留再获益，只能要求不退化，其他数量点另作诊断。逐视频变化、配对翻转及分母必须同时披露，不以pooled指标掩盖real失败。

**未通过如何收束：** R失败，说明本核心监督/预算尚未形成足够参考范围收益；R通过而Q1失败，说明参考更准仍未转化为所需错误排序。两者均不认定唯一根因，不在原结果上反号/放宽门槛/延长轮数/挑epoch/按视频回退或转向方向结构；记录此次有限负结果并保留原simple。R与Q1通过但Q2仍有代价，说明95%工作点尚不合适，另立版本确定正确保留/误拒约束，而不是加大模型期望突破接受数的组合下界。如需其他预算或机制，应另行记录版本与依据，不覆盖本次预设条件。

### 21.7 本轮交付与后续边界

已完成上下文恢复、相关直接源码静态核对与上述单一设计记录。公式的曲率—尺度关系是解析推导；新增loss尚未实现，数值/梯度、同初始化控制、真实GPU保存重载、训练、参考效果和新标志判断全部未执行，不能写成已通过。当前不提供可执行新训练命令，避免把未实现设计伪装成可运行版本；后续获授权后再本地实现、必要验证与审查，并给用户服务器指令。

现有TRAIN/VAL及其多次研究记录属于开发证据，不能称新独立确认。**TEST已多次暴露，本设计不用TEST的错误分布、视频、成绩、门限或候选选择；本轮没有再次读取原始TEST产物。** 第18节用于认清冻结基线身份和保留已核验结论，未用其计数制定上述门槛。中心保持覆盖，方向暂不推进；等比例变换、网格/原图链、宽高交换/π周期及深度接口约束继续保持，接口保持不代表已验证深度/物理安全精度。所有记录集中本文，不更改几何窗口、大小论文或旧模型/结果身份，不新增本地回传副本、不连接服务器。

### 21.8 用户补充与有限实现（2026-10-04；真实服务器预检待运行）

**新增授权：** 用户认可有限设计，要求补充实际梯度强度与95%覆盖上限后，实现有限TRAIN预检；预检通过后开展两臂固定对照。此前21.1/21.7的“未实现”描述保留为设计阶段事实；当前代码状态以本节为准。不连接服务器，实际GPU检查/训练由用户运行；几何窗口、大小论文、原policy/权重/结果和已有改动保持。

**已实现，尚未运行真实训练：** 新增以下8个文件，父入口及父来源清单不修改：

- [统一入口](../crane_project/tools/run_port_size_core_curvature_v2.py)：`check → smoke → train → assess`，分别新建目录；旧失败目录、smoke权重、其他epoch/arm或不同合同不能继续。
- [NumPy固定GT投影与独立解析梯度](../crane_project/utils/port_size_core_curvature_v2.py)、[Torch损失与梯度测量](../crane_project/utils/port_size_core_curvature_v2_torch.py)：保持原参考架构/像素项；新增核心曲率与二次残差。GT投影使用float64 QR，实际Torch的辅助项导数另与独立NumPy公式逐图比较。
- [配对固定评价入口](../crane_project/tools/eval_port_size_core_curvature_v2.py)、[比较与分母统计](../crane_project/utils/port_size_core_comparison_v2.py)：复用精确pin的正式midpoint/原simple/固定模板链，在432参考留出＋全部887 VAL上比较A0/A1；不使用旧B矩读取来替代。只产生数值证据和比较工作点，不产生最终部署policy。
- [新协议](../crane_project/tools/port_size_core_curvature_v2_protocol.json)、[新来源清单](../crane_project/tools/port_size_core_curvature_v2_sources.json)、[必要测试](../tests/test_port_size_core_curvature_v2.py)。来源清单锁定本版7个直接文件及父模板来源；不重写旧来源身份放行。

**实际梯度强度预检：** 在原14张拟合角色真实图上，两臂保持完全相同且未更新的初始化；记录`L_v1/L_curv/L_quad`各自参数梯度原始范数、乘1/.25/.05后的范数、合成新增梯度范数及其相对原项比、二者方向余弦，stem/output分组也单独记录。总梯度使用共同clip10，记录裁剪比例及各加权项按同一比例投影后的范数；这不是分别裁剪每个loss。随后每臂4个real/sim交替真实更新，验证实际backward与分项梯度合成一致、实际裁剪前后变化、原B/midpoint输出及冻结状态保持、保存重载一致，smoke模型丢弃。

在任何真实新梯度结果前固定宽泛**工程防失衡护栏**：real/sim各自合格初始视图的`||.25g_curv+.05g_quad||/||g_v1||`中位数在[.01,10]，共同clip保留比例中位数≥.1；两域必须有合格记录，新增梯度必须实际到达stem/output，所有分项/合成/实际梯度有限。护栏不从loss值推断系数合适，不证明系数最优或训练有效；完整逐图值、零quad梯度及抵消情况仍保留。护栏失败记录并停止，不现场调整.25/.05、护栏或clip后直接重跑。正式训练每64个slot测一次全部分项梯度，其余步仍记录三项loss及实际总梯度裁剪，所有1536步都更新两臂，不因是否测诊断而跳步。

**比较、最终标志与论文身份：** 95%及同接受数只判断固定覆盖下的参考精度/错误排序/观测取舍，报告自身支持、共同支持、全输出和每组FA理论下界；最终工作点等待排序收益成立后，按正确保留/误拒约束另立TRAIN/VAL版本。参考更准、真实判断排序更准、最终工作点更合适分别下结论。曲率监督及“恢复参考长短边再比较”均非PQA原方法；共享冻结P3不保证参考与检测错误独立。本候选不接DINO、蒸馏、候选排序或方向结构。

**本地验证（已测；不等于服务器预检）：** 使用现有已核验TRAIN输入snapshot在内存恢复原384/432/32角色；只对原14个拟合视图计算新增loss数值机制，14/14核心合格且通过。理想K误差最大4.4409e−16，±15%四类尺寸扰动的最大方向有限差分相对误差9.5047e−9；它们是解析Gaussian/GT几何上的新loss检查，不是图像学习或真实网络梯度强度。未落盘新的回传副本或热图，不把旧理想读取当新收益。最终17项必要unittest中15项通过、2项真实Torch自动微分/参数梯度/保存重载测试因本地无Torch明确跳过；服务器必须执行后两项。通过项覆盖平方尺度关系及双侧梯度、曲面残差独立有限差分、等比例/反射/π与宽高等价、padding与小尺寸跳过、GT不入在线API、95%组合下界、同分边界、无输出/参考不可用/共同集合分母、无标签比较门限、分域/逐视频数值JSON重放、失败阶段/身份保护、源码和Python3.8静态兼容。模拟优化器调度检查验证两臂全部1536步都更新、同图同序、非诊断步不会漏更新；这是流程测试，不是真实参考训练。命令行`--help`通过。新增入口/损失/读出及阶段合同已静态复核，原四份父直接源码清单共18项仍相符；本地未重载原GPU权重或再次确认完整父缓存。不能以本地跳过状态放行真实训练。

**共享工作区记录：** 期间出现提交`78a2fbf`，其差异包含几何窗口内容及部分尚在验证的可靠性新增文件；本窗口没有执行提交、回退或修改几何代码。最终新来源清单按当前完整实现刷新，不能用该提交中途收录的旧新版本SHA替代最终清单；各冻结父来源保持原字节。已有改动/提交均保留，不处理其他窗口工作。

**服务器运行顺序（仅给指令，本地未执行；所有输出使用新目录）：** 上传上述8个新文件到对应相对位置，保留旧源码/来源和全部固定输入。沿用此前GPU3示例，进程逻辑GPU为0：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -m unittest discover -s tests -p 'test_port_size_core_curvature_v2.py' -v

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_core_curvature_v2.py \
  --mode check --out-dir work_dirs/port_size_core_curvature_v2_check

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_core_curvature_v2.py \
  --mode smoke --gpu 0 \
  --check-report work_dirs/port_size_core_curvature_v2_check/check_report.json \
  --out-dir work_dirs/port_size_core_curvature_v2_smoke

# 仅在真实单测无跳过、check与smoke成功且实际梯度报告满足固定护栏后继续。
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_core_curvature_v2.py \
  --mode train --gpu 0 \
  --check-report work_dirs/port_size_core_curvature_v2_check/check_report.json \
  --smoke-report work_dirs/port_size_core_curvature_v2_smoke/smoke_report.json \
  --out-dir work_dirs/port_size_core_curvature_v2_train

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/run_port_size_core_curvature_v2.py \
  --mode assess --gpu 0 \
  --train-report work_dirs/port_size_core_curvature_v2_train/train_report.json \
  --out-dir work_dirs/port_size_core_curvature_v2_assess
```

放行终态分别为`CORE_CURVATURE_NUMERICAL_CHECK_PASS`及`CORE_CURVATURE_TRAIN_PRECHECK_SAVE_RELOAD_PASS_DISCARDED`；正式两臂必须各1536步、同初始SHA、固定epoch04，保存重载必须通过。训练完成和评价完成的`REVIEW_REQUIRED`只表示工程完成，不是R/Q1收益通过或最终标志已可用。不读TEST、不依据其多次暴露内容调参/选权；不修改原95%基线policy或已有1440帧结果。

### 21.9 服务器TRAIN梯度核对失败与数值修复（2026-10-04）

**用户日志事实：** 旧入口已进入真实服务器配对TRAIN，打印epoch1的1/65/129/193/257/321号slot后，诊断更新处报`Component decomposition differs from actual backward`。日志未给出出错arm、确切slot/图像或误差幅度；不据此断言服务器实际损失发散、完成了某个epoch，或唯一原因为梯度抵消/TF32。旧单测与smoke完整产物尚未回传核验。服务器traceback行号与当前本地入口不完全一致，更新时发送全部8个候选文件保证版本配套，不重写冻结父来源。

**本地已复现的检查缺陷：** 对同一共享两层Linear图的反向传播，构造几乎抵消的v1与.25curv梯度，参数分项合成与一次总loss backward的最大差异为1.86188e−6，旧规则只允许1.24319e−9，因其容差依赖接近零的合成结果，会把正常FP32计算顺序差异拒绝。[PyTorch数值说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)确认数学等价的浮点计算不保证逐位相同，TF32还会减少卷积/矩阵乘法输入精度；它支持工程处理，不证明本次服务器唯一原因。

**有限修复：** 每个参数tensor分别检查L2和最大误差，尺度改为`sum_k ||c_k g_k||`及`sum_k max|c_k g_k|`，在抵消前衡量；固定相对容差`128 × dtype_epsilon`、绝对底限1e−12。FP32相对容差约1.52588e−5，不能由其他大tensor掩盖某个小tensor缺失/符号或系数错误；缺失、非有限、shape/dtype不一致及超限仍在optimizer.step前停止。实际裁剪保留比例改用实际backward梯度范数作分母。分项与实际梯度核对完整记录逐tensor差异、尺度和限额；超限`failure.json`增加梯度详情及epoch/slot/图像/域/arm上下文，不再只给一句错误。

候选参考的前向与所有反向传播（包括非诊断训练步）在局部上下文中关闭matmul/cudnn TF32，随后恢复原标志；冻结B/midpoint的前向及几何读出算法沿用原设置，不改变全局推理精度。[PyTorch1.8卷积自动微分实现](https://raw.githubusercontent.com/pytorch/pytorch/v1.8.0/tools/autograd/derivatives.yaml)会将前向`allow_tf32`传给卷积backward，因此只在调用backward时切标志不足；新`SizeReference`只包装原forward的精度上下文，继承原架构/参数名/初始化。相同精度处理应用于两臂，保存重载及配对评价也使用同一候选参考包装。这是候选参考数值协议修复，不是性能选优；原1/.25/.05损失、Adam/clip10、种子、1536步、参考架构、角色和R/Q1/95%职责不变。协议记录`gradient_decomposition_check`，入口核对该字段，新来源SHA已刷新，旧check/smoke合同不能释放新版train。

**修复版本地验证：** 新发现本机已有`/opt/anaconda3/envs/mmrot/bin/python`（Python3.8/PyTorch1.8.0.post3/NumPy1.21.3）及`/opt/anaconda3/envs/yolo/bin/python`（PyTorch2.9.1/NumPy2.2.6），均无CUDA可用；未安装新依赖。因此21.8“本地无Torch”保留为当时使用的bundled环境事实，现已在这两个已有环境分别运行21项测试，全通过、无跳过。覆盖真实Torch辅助解析梯度、抵消误判回归、主动注入错误分解时拦截optimizer、逐tensor容差/零梯度/非有限检查、精度标志在嵌套与异常时恢复、参考架构与初始状态逐项相同且实际Conv2d forward使用关闭的TF32设置、真实参考更新/裁剪/保存重载、完整3072更新与失败上下文、原比较协议。测试在有CUDA的服务器也自动运行抵消/错误分解及真实候选卷积精度GPU用例。本地验证不等于正式冻结特征GPU预检，不含真实候选训练或性能测量。

**新版运行与回传：** 保留旧`check/smoke/train`目录及失败证据，不续训失败权重；新版统一使用`port_size_core_curvature_v2_{check,smoke,train,assess}_gradfix1`四个新目录。先新版单测无跳过，再重新check/smoke，满足工程护栏后按同初始化/固定epoch04从头训练两臂、最后TRAIN留出＋VAL评估；不能沿用旧smoke报告或放宽原强度护栏放行。服务器运行次序与21.8相同，仅四个目录加`_gradfix1`并对应修改报告路径。分析包保留旧失败train目录和已生成的新版四目录，排除`*.pth`、保留SHA标记与所有JSON/JSONL；若中途失败只打包已存在的目录。不连接服务器，不复制本地回传结果，不改基线或TEST。用户要求后续此类交付仅给运行指令与分析结果压缩指令。

### 21.10 gradfix1第二次失败：同上游梯度核对与失败帧预检（2026-10-05）

**用户日志事实：** gradfix1已经完成epoch1并到epoch2 slot1，`real_seq05_00142`、A1、`output.weight`的参数分项合成最大差异1.28801912e−6，限额9.02819731e−7；L2差异5.89720751e−6，限额5.07307748e−6。两者分别超过约1.43/1.16倍，说明21.9修复不足。不能由此认定新监督已无效或损失发散，也不能仅以小绝对差异认定服务器梯度正确。实际失败文件/权重尚未本地读回；本次依据用户日志与已交付源码修复，保留原失败证据。

**修复方法与职责：** 不再把3次独立上游种子的卷积参数反传相加当作一次总loss backward的严格等式参考；128ε没有提供任意空间归约长度下的误差保证。保留三项实际参数梯度范数、加权尺度、方向、共同裁剪及原参数分解差异，原差异明确标为`diagnostic_only_different_convolution_backward_seeds`，不能将其`passed`字段当成训练放行或性能条件。

真实训练仍执行原总loss的`total.backward`。新增两项硬检查：（1）在共享热图logits上核对总loss实际反传的上游梯度与三项loss各自导数乘固定系数后的和；（2）捕获该次反传真实上游梯度，使用完全相同的梯度作为`grad_outputs`，通过同一logits→参数图做一次VJP，逐参数核对实际`.grad`。数学职责参考[PyTorch autograd.grad](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html)，它计算给定上游向量的VJP且不累积到参数`.grad`。共享logits检查与同种子VJP逐tensor检查仍用固定128ε＋1e−12，未扩大容差来放行错误。参数缺失、非有限、shape/dtype不符、不一致或无效clip均在optimizer.step前停止，完整失败上下文保留。

候选参考继续局部关闭TF32，并同时设置cudnn deterministic=True、benchmark=False以固定重复卷积反传算法；参考前向/反向/保存重载/评价统一该上下文。退出时恢复全部4个原标志，冻结B与正式midpoint沿用原设置。相关工程职责见[PyTorch可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。这不是跨设备逐位一致承诺，也未在本机无CUDA条件下认证服务器GPU实现。原架构、初始权重、损失1/.25/.05、Adam/clip10、1536步与TRAIN/VAL/TEST职责不变，两臂使用同一执行协议。实际更新记录改用实际backward梯度的裁剪前范数、实际共同裁剪保留比例，三项/新增项的裁剪后范数仍是按该共同比例投影，另保留参数分项合成的预测范数，不能混作逐项独立clip。

**新增有限`replay`阶段（工程预检，不训练、不续训）：** 新check及smoke通过后，从保留的`port_size_core_curvature_v2_train_gradfix1/a1/epoch_01.pth`读取原384步A1状态，检查checkpoint/marker SHA、原合同/role/epoch/steps、初始SHA和冻结前端。原manifest SHA固定`1bd91bd6b4696b1932915750a961c721567becc575143d327387003a7efe9f28`，原7项源码表fingerprint固定`de0359c76d50c2680587f0ebaf0e3a5b9c36bfb157477b45f06cda72779ae276`，原protocol fingerprint固定`e99467c8ac340bad602f200aaeea13c2b7fa9777ff535af8f401a05a44b36e4c`；其余冻结输入与新合同完全相同，不放宽父来源pin。

只按原384 TRAIN拟合列表、RandomState(1703)的第一项读取上述失败图，核对原failure.json的epoch/slot/图像/域/arm；验证B/midpoint输出与原collect缓存一致。执行当前梯度与clip检查后，InspectOnly优化器仅计调用而不更新参数，要求checkpoint文件、参考参数与冻结输出保持。终态为`CORE_CURVATURE_FAILURE_FRAME_REPLAY_PASS_NO_OPTIMIZER_UPDATE`，写`replay_report.json`完整梯度、原失败/输入/checkpoint及marker身份。它只证明该失败帧的工程路径，不用于阈值或候选选优；不加载旧optimizer，不将384步计入新训练。新版train必须带同合同的replay报告并从原同初始化重训两臂。缺旧文件或身份不符即停止，不删除failure.json或合成完成标志绕过。

**本地验证已完成，真实服务器尚待：** Python3.8/PyTorch1.8与PyTorch2.9两个已有CPU环境各23项必要测试全通过、无跳过。包括共享logits导数故意篡改拦截、同种子参数VJP故意篡改拦截、抵消回归、原独立NumPy解析导数、真实参考更新/裁剪/保存重载、4个精度/算法标志嵌套及异常恢复、旧来源/合同及角色保护、模拟旧epoch1文件的真实Torch失败帧预检不更新参考参数或checkpoint、全部3072优化器更新及原覆盖比较职责。模拟旧权重不是实际服务器epoch1，不形成实际失败帧已通过或新尺寸性能已改善的证据。服务器单测在有CUDA时也执行梯度错误注入和参考卷积用例。未连接服务器，未读取TEST或新增本地回传副本，几何窗口/实际论文保持。

**当前运行顺序与交付格式：** 上传8个配套文件的`port_size_core_curvature_v2_gradfix2_20261005.tar.gz`到服务器项目`work_dirs`，保留原train及gradfix1失败目录。新目录使用`port_size_core_curvature_v2_{check,smoke,replay,train,assess}_gradfix2`；顺序为单测无跳过→check→smoke→replay→train→assess，train增加`--replay-report`，replay增加`--failed-train-dir work_dirs/port_size_core_curvature_v2_train_gradfix1`。环境设置可以合并并显式加分号防粘贴丢换行；每个阶段命令单独一条，不将全部任务包成一个shell执行块。分析结果在项目`work_dirs`内压缩为单一tar.gz，收集已存在的旧失败train、gradfix1失败train、新单测日志及5个阶段目录，排除`*.pth`但保留SHA标记与JSON/JSONL。当前未声明任何R/Q1收益或最终标志已可用。

<a id="legacy-35"></a>
<a id="legacy-36"></a>
<a id="legacy-37"></a>
<a id="legacy-38"></a>
<a id="legacy-39"></a>
<a id="legacy-40"></a>
<a id="legacy-41-structure"></a>
<a id="legacy-43"></a>
<a id="legacy-46"></a>
<a id="legacy-47"></a>
<a id="legacy-r1"></a>
<a id="current-r123"></a>
<a id="simple-r124"></a>
<a id="simple-r125"></a>
<a id="simple-r126"></a>
<a id="shi-r127"></a>
<a id="scope-r128"></a>
<a id="closure-r129"></a>
<a id="closure-r130"></a>

[替换前完整可靠性交接（原编号/原锚点）](archive/20261003_reliability_replaced_by_simple_v1/readable/docs/reliability_handoff_20261003.md)；[归档完整性清单](archive/20261003_reliability_replaced_by_simple_v1/manifest.json)。


### 21.11 gradfix2服务器回传：工程通过，参考与判断收益未通过（2026-10-05）

**范围与结论（事实）：** 用户提供终端文本和`port_size_core_curvature_v2_analysis_gradfix2_20261005.tar.gz`，包SHA `a7344d4638601817271ffeb2eb10d2da49f73076284b0fe48016b023ab8d9d72`。此次已经完成有限TRAIN预检、原失败帧复核、两臂固定4轮训练及432参考留出＋887全量VAL评价，不能再把正式train/assess列为待运行。E工程门槛通过；第21.6节预设R参考精度与Q1真实判断门槛均未通过。本候选不替换冻结simple、不进入最终工作点选取，不据此启动TEST。新增监督有局部效果，不能概括为没有梯度或全部无效；也不能把局部效果称为可靠性优化成功。

**核验范围（事实及限制）：** 只在内存读取用户回传，没有解包或新增本地结果副本，没有连接服务器、重新训练/推理、打包本地源码或修改算法。5阶段completion索引的13个产物字节SHA均匹配，合同/直接来源与当前代码一致，B24、midpoint23和simple policy仍为第20.3节身份；没有新失败文件。1319条逐帧记录重新计算全部参考精度、排序、混淆、工作点和汇总，结果精确一致。终端与包内23项单测全部通过，无跳过。包排除了权重，仅有SHA标记，故本地未重新哈希实际`.pth`字节或重做GPU保存重载；保存重载精确一致是已核对来源和产物身份的服务器报告证据。基线原policy/权重/门限及TEST结果未更改。用户明确后续源码通过Git同步，不再生成本地源码压缩包；服务器分析回传仍使用项目`work_dirs`下单一压缩文件。

**工程阶段与梯度（事实）：** check为`NUMERICAL_CHECK_PASS`，smoke为`TRAIN_PRECHECK_SAVE_RELOAD_PASS_DISCARDED`；14个拟合视图×两臂同初始化测量，随后各4步smoke并保存重载一致，smoke参数丢弃。原失败帧`real_seq05_00142`在原gradfix1 A1 epoch01权重复核成功，optimizer更新0、参数不变；同上游共享logit与参数VJP核对通过。旧的独立分项参数梯度相加仍精确复现`output.weight`误差1.28801912e−6/L2 5.89720751e−6，其`passed=false`是保留的诊断，不是新核对失败。新方法未扩大原容差。

正式两臂同初始状态SHA `d6b3743da3edad8901aaa1c025780141fa4272817f849e9307385246ba71886b`，各1536步、合计3072次更新，固定epoch04。日志完整，每轮各384唯一图像，两臂每步图像/顺序/投影资格相同；所有拟合图均可用新增监督。48个预设分项梯度测量步的新核对全部通过，其他步保留原总loss反向、有限数值和实际裁剪记录，不能称3072步均执行了分项核对。所有loss有限；A0裁剪前最大范数9.86263，A1为10.03017，仅A1一步超过clip10，最小实际范数保留率0.996992，其余约1，未发生大幅裁剪。

| 同初始化A1域/视图数 | 原项梯度范数中位 | 曲率原始→加权中位 | 二次残差原始→加权中位 | 新增合成/原项中位 | 预检裁剪保留 |
|---|---:|---:|---:|---:|---:|
| real / 10 | .152297 | .353491→.088373 | .000652→.0000326 | .565413 | 1.0 |
| sim / 4 | .158939 | .393629→.098407 | .000844→.0000422 | .607414 | 1.0 |

二次残差加权梯度/原项中位仅real .0215%、sim .0264%，新增作用主要来自曲率。训练抽样测量的A1新增/原项中位逐轮real约1.19/1.89/2.91/3.55、sim .65/4.53/3.78/3.06；sim最后一轮抽样夹角余弦中位−.584，只有2个sim测量点，不能认定全程冲突或唯一根因。这些是强度事实，不证明`.25/.05`最优；本次不改系数、不搜索梯度比率。

**参考精度（事实）：** 下表均在A0/A1共同可用输出上，误差为两条规范边的最大相对误差；10%正确覆盖最后一列以该组全部检测输出为分母，计入不可用代价。TRAIN留出各域分别只有一条视频，故域级数值即`real_seq13/sim_seq08`；sim VAL只有`sim_seq10`。

| 组 | 共同输出 | 均值 A0→A1 | P90 A0→A1 | 两边10%内数 A0→A1 | 全输出参考正确覆盖 A0→A1 |
|---|---:|---:|---:|---:|---:|
| TRAIN留出 real/seq13 | 304 | 19.8606%→11.9912% | 29.6883%→19.7241% | 19→121 | 6.2500%→39.8026% |
| TRAIN留出 sim/seq08 | 126 | 26.9361%→39.9636% | 63.3803%→106.1925% | 46→45 | 35.9375%→35.1562% |
| VAL real | 372 | 18.9709%→17.3147% | 27.6367%→25.2851% | 65→81 | 17.3797%→21.6578% |
| VAL sim/seq10 | 511 | 16.9164%→15.9693% | 31.6336%→32.1522% | 205→198 | 40.0391%→38.6719% |
| VAL real_seq07 | 223 | 18.7691%→20.7374% | 27.1112%→28.4701% | 48→23 | 21.3333%→10.2222% |
| VAL real_seq14 | 149 | 19.2730%→12.1921% | 28.1780%→18.7138% | 17→58 | 11.4094%→38.9262% |

real VAL均值只相对降低8.7302%（要求≥20%），共同支持10%内比例17.4731%→21.7742%，增加4.3011个百分点（要求≥10）；全输出覆盖只增加4.2781个百分点。seq14正确数17→58，但seq07为48→23，其全输出参考正确覆盖21.3333%→10.2222%。sim VAL P90微升，正确数205→198；sim参考留出均值/P90明显增加。可用数TRAIN real304→304、sim128→126，VAL real373→372、sim512→511；TRAIN两项新增不可用都是正确检测，VAL不可用A0为1坏、A1为3坏，不可用不能从全输出评价中消失。GT aspect探针共同支持分母real VAL372、real留出304；长/短双侧辨识A1分别41.3978%/29.3011%、50.9868%/35.1974%，均未满足各边60%。因此R明确失败，不能仅以real均值下降宣布通过。

**同接受数判断（事实）：** 每格顺序为`FA/FR/ED/CR`＝错误接受/正确误拒/错误检出/正确保留。接受数由原冻结simple预先规定；各组分别排序匹配，不能把逐视频排序结果相加冒充域级排序。主表均可达，TRAIN sim A1例外，只能接受126个。域/视频对应关系同上，全部方法保持原检测输出分母。

| 组 | 接受/总帧（覆盖） | score | simple | A0 | A1 |
|---|---:|---:|---:|---:|---:|
| TRAIN留出 real/seq13 | 257/304 (84.5395%) | 2/45/2/255 | 3/46/1/254 | 2/45/2/255 | 2/45/2/255 |
| TRAIN留出 sim/seq08 | 128/128 (100.0000%) | 0/0/0/128 | 0/0/0/128 | 0/0/0/128 | 0/2/0/126 |
| VAL pooled | 843/887 (95.0395%) | 216/15/28/627 | 212/11/32/631 | 217/16/27/626 | 219/18/25/624 |
| VAL real | 333/375 (88.8000%) | 159/12/29/174 | 158/11/30/175 | 166/19/22/167 | 160/13/28/173 |
| VAL sim/seq10 | 510/512 (99.6094%) | 54/0/2/456 | 54/0/2/456 | 55/1/1/455 | 54/0/2/456 |
| VAL real_seq07 | 196/226 (86.7257%) | 160/8/21/36 | 154/2/27/42 | 162/10/19/34 | 162/10/19/34 |
| VAL real_seq14 | 137/149 (91.9463%) | 6/11/1/131 | 4/9/3/133 | 6/11/1/131 | 7/12/0/130 |

TRAIN sim A1实际接受126/128=98.4375%，其表格`0/2/0/126`不是128同数成功。TRAIN real仅4坏、sim0坏，不把留出real AUROC .806667作为跨域检错成立证据。

| VAL共同支持组 | 输出/坏 | score AUROC | simple | A0 | A1 |
|---|---:|---:|---:|---:|---:|
| real | 372/186 | 0.707249 | 0.589085 | 0.373772 | 0.548676 |
| sim/seq10 | 511/55 | 0.626077 | 0.695215 | 0.682616 | 0.694019 |
| real_seq07 | 223/179 | 0.438167 | 0.580879 | 0.419502 | 0.512697 |
| real_seq14 | 149/7 | 0.649899 | 0.753521 | 0.490946 | 0.430584 |

real主点A1较A0少6FA/6FR，但较simple多2FA/2FR；预设要求不多于153FA，实际160。A1 real AUROC较A0提高.174905，但仍低于simple及score，未达到score .707249＋.02。seq07比simple多8FA/8FR，seq14多3FA/3FR；两条视频均不满足不劣条件。sim主点与simple相同，不能抵消real失败。Q1明确失败。

**固定工作点与取舍（事实）：** pooled95%比较接受843/887=95.0395%，642好输出决定FA下界201；simple212、A0 217、A1 219。A1多7FA且多7FR，并不是95%限制导致它必然比simple差。全局A1比较门限的实际组接受数为real357、sim486、seq07 208、seq14 149；本门限只用于本次比较，不是部署policy。下表在各组A1实际接受数匹配另外三种排序。

| A1实际数量组 | 接受/总帧（覆盖） | score FA/FR/ED/CR | simple | A0 | A1 |
|---|---:|---:|---:|---:|---:|
| pooled | 843/887 (95.0395%) | 216/15/28/627 | 212/11/32/631 | 217/16/27/626 | 219/18/25/624 |
| real | 357/375 (95.2000%) | 173/2/15/184 | 172/1/16/185 | 177/6/11/180 | 177/6/11/180 |
| sim/seq10 | 486/512 (94.9219%) | 49/19/7/437 | 46/16/10/440 | 41/11/15/445 | 42/12/14/444 |
| real_seq07 | 208/226 (92.0354%) | 166/2/15/42 | 164/0/17/44 | 170/6/11/38 | 170/6/11/38 |
| real_seq14 | 149/149 (100.0000%) | 7/0/0/142 | 7/0/0/142 | 7/0/0/142 | 7/0/0/142 |

相对冻结simple实际规则，A1 real CR175→180、FA158→177，sim CR456→444、FA54→42；接受数改变，不能只用sim少12FA称全面改善。全局A0组工作点另为real356（176/6/12/180）、sim487（41/10/15/446）、seq07 207（169/6/12/38）、seq14 149（7/0/0/142）。按每组自身95%数量的次级诊断，real接受357，score/simple/A0/A1 FA分别173/172/177/177；sim接受487为50/46/41/43。90%诊断real接受338，FA162/161/168/164；sim461为47/43/40/38。sim低覆盖点有局部排序收益，A1在95%仍差于A0，不能挑90%替换预定主点。全部逐视频90/95%、共同支持、并列界及A0实际数匹配细表继续以包内`assessment.json`为原始证据。

**中心和隔离（事实）：** VAL real输出374/375=99.7333%，输出帧中心正确362/374=96.7914%，全帧正确中心362/375=96.5333%；real_seq07为225/226、213/225、213/226，real_seq14三项均149/149。sim三项均512/512；TRAIN留出real/sim三项均304/304、128/128。前端、候选参考评价前后状态不变，正式B/midpoint原框差最大0，原框/score/输出数/中心与方向标志全部保持，GT_online=false、TEST_read=false。保留正式midpoint原已核验定位、sim角度、两域RIoU收益，不把本次尺寸参考负结果归因于正式midpoint没有效果。中心仍为保留有效输出，未新建中心正确性门控。

**新增有限行级观察（事实，不是完整诊断）：** 从本包既有参考/GT边长比，在共同支持上读出中位数。real_seq07长/短比A0 .9447/.9001→A1 .8532/.8866，原已偏小进一步偏小；seq14为1.0856/1.1783→.9693/1.1092，减小缓解原偏大。TRAIN real_seq13为1.0538/1.1871→.9253/1.0917；TRAIN sim_seq08长边比P90 1.6338→2.0619，虽中位数1.0420→1.0116，尾部仍恶化。参考自身准确不保证错误排序：seq14参考正确数增加41，但该视频A1 AUROC .430584低于A0 .490946和simple .753521。

**下一步建议（推断/尚未实施）：** 收束本次固定曲率版本为“工程可行、局部参考改善，但R/Q1未成立”，保留A0/A1身份、固定epoch04、原门槛与全部负结果。当前不延长训练、调`.25/.05`、反转风险、挑epoch或修改最终工作点，也不转向方向结构。下一项优先设计与固定半峰模板读取更一致的**长短轴内/外范围监督**：用GT固定采样位置的幅值/范围约束，直接检验同一机制能否拉回偏大与偏小，而不只要求GT核心拟合曲率接近I；图像分支、在线读取器、前端和判别对照保持。现有结果支持“普遍收缩对不同原始偏差产生不同收益”及“曲率代理与在线读出对齐不足”作为待验证机制，但尚未证明读出不一致或梯度冲突为唯一根因。若开展，先另立一项有限设计并固定公式、系数、同初始化预算、TRAIN拟合/留出职责、双侧探针及R/Q1继续条件，再获授权实现，不在本次模型上补调。即使参考更准，仍必须独立证明真实排序收益与原simple正确观测保护；R/Q1成立后才研究受正确保留/误拒约束的最终工作点。所有后续选择继续只用TRAIN/VAL，TEST已多次暴露、不用于调参或重选。


## 22. 固定半峰读取的长短轴内／外范围监督v3（2026-10-05）

### 22.1 授权、已知问题和有限新设计

用户授权收束第21节曲率v2、按21.11建议设计并实现下一项有限监督、必要本地验证、提供服务器运行及结果压缩指令。曲率v2固定epoch04和全部负结果原样保留；不覆盖任何旧源码/清单/模型/结果，不沿用旧头继续训练。Git负责源码同步，不再打包本地源码。范围仍仅为尺寸参考监督，前端固定B24＋正式midpoint23＋现有simple，不修改框/score/输出数、中心或方向，不修改几何窗口或实际论文稿、不读取TEST或连接服务器。TEST已经多次暴露。

**依据（事实）：** v2使real_seq14参考10%内17→58，却使seq07为48→23；两视频参考边长中位均缩小。sim参考留出长边比P90 1.6338→2.0619。real主同数A1 FA160仍高于simple158，R/Q1未成立。局部参考精度改善不代表真实判断成功，现证据未证明唯一根因。

**机制假设（待验证）：** 用与半峰范围直接对应的内侧、边界、外侧响应约束，替代仅要求GT核心二阶曲率接近I；同一机制必须能纠正偏大和偏小。继续保持图像分支、在线读出、风险和评价不变，仅改训练辅助项，不扩充质量分类器。

### 22.2 固定公式与可微实现（先于新训练冻结）

规范GT长短边L/S与角θ经原等比例/反射坐标链变换到模型空间，σ_L=L/4、σ_S=S/4。每条轴取两个正负方向、横向sigma偏移t∈{−.25,0,.25}，半峰半径h(t)=sqrt(2 ln2−t²)，沿轴位置为q h(t)，q∈{.85,1,1.15}。故理想同轴Gaussian在q=1处相对中心响应为.5；q=.85为内侧，q=1.15为外侧。外侧指半峰边界外，仍位于GT OBB内；GT OBB外的背景另由背景集合约束。长短轴分别18点，共36点；另固定GT中心一点。所有位置由GT和变换几何决定，不依预测响应或错误标签选点。

采用现有stride2网格(j+.5)×2−4，即原点−3，固定4邻点双线性采样B_i。训练背景集合是GT上下文内、有效且原GT target=0的网格：上下文边长仍max(3L,64px原图)，背景取当前预测p的Torch lower median b；中心p_c=B_c(p)。预测相对响应 r_i=(B_i(p)−b)/max(p_c−b,.05)，GT目标 r_i*=B_i(T_GT)/B_c(T_GT)。目标使用同一双线性算子抵消网格相位偏差，所以边界目标仅近似.5，数值门槛为最大偏差≤.03，而非声称离散地图有精确半峰。GT target是OBB派生Gaussian，不是可见轮廓标注。

L_inner=mean_{两轴×正负×横偏×q∈{.85,1}} SmoothL1_.1(r_i−r_i*)；L_outer相同但q∈{1,1.15}，各24项。长短轴、正负方向和横偏数量相等；边界q=1在两项中各占一半。A0只有原L_v1；A1固定L=L_v1＋.125 L_inner＋.125 L_outer，等价总范围权重.25下平均两项。原soft focal BCE像素项、Adam lr.001/wd0/clip10不变。系数不由新VAL成绩或测量梯度回搜；若工程强度门槛失败先停止，不自动缩放权重。

预测p=sigmoid(logits)、双线性采样、背景median、中心和强度分母均保留Torch梯度；GT采样索引/权重/目标为固定NumPy常数。不截断r_i为[0,1]，低于背景的样本也能收到梯度；只有低对比分母使用预先固定.05保护，其上下分支均做实际有限差分核对。median/floor分段可微；有限差分用唯一中位数且远离排序/保护拐点的受控图，避免大背景集合跨排序边界造成伪失败，不修改实际图像loss。梯度核对继续使用v2已通过的共享logit及同上游种子VJP，候选参考局部关闭TF32并使用确定性卷积，随后恢复原标志。旧v2文件不改。

**实现边界（必须披露）：** 训练用GT中心、GT背景区域，在线模板用检测框上下文、预测峰值、硬半峰连通核心和IRLS。新监督仅对背景扣除、相对强度和半峰内/外范围更直接，并非将在线模板变为可微、也不保证两者相同；GT不进入在线API。该项目适配并非PQA原方法，共享冻结P3不证明误差独立。短边模型尺寸<16px、背景不足16格、任何采样邻点越过有效图像或中心GT响应不可解析时，仅跳过新增项并保留原像素项；不依据真实正确性选帧。原因/数量随check、TRAIN及离线评价报告，不剔除全输出帧。

### 22.3 对照、角色和继续条件

两臂重新同seed1701初始化、同384拟合张、同每轮seed1701+epoch顺序，4轮各1536步，固定epoch04，不选epoch、不续训v2；smoke各4步后丢弃。沿用432参考留出（real304/sim128）、32 guard；拟合角色决定TRAIN数值检查和14视图梯度预检，留出/VAL只用于固定epoch04离线精度和判断检验，不拟合loss/权重或风险模型。检测器/simple已见检测TRAIN，留出仅对参考分支成立，不称全系统未见。

E：理想离散GT图范围项近零、±15%长短边各两侧梯度方向正确；真实Torch每项有限、stem和output三项梯度非零；两域初始化新增/原项范数中位在[.01,10]、裁剪保留中位≥.1。完整记录三项raw/weighted范数、新增比率/夹角、裁剪前后及shared-logit/VJP核对。原检测/midpoint状态和输出不变、保存重载精确一致。此为宽工程门槛，不证明系数最优。数值check通过不替代服务器GPU smoke。

R/Q1/Q2沿用第21.6节全部原预设门槛，不因本次结果改变：real共同支持均值至少相对降低20%、P90不升、共同/全输出参考正确覆盖均至少+10个百分点；sim VAL和参考留出两域均值/P90不升、全输出正确覆盖不降，各域可用数不降，逐VAL视频覆盖不降；real VAL/留出长短边±15%双侧探针各≥60%且不低于A0。Q1主接受数real333/sim510及原每视频simple接受数；real A1较score/simple/A0最好者FA至少少5、sim不劣，每视频不劣score/simple较好者且至少一real视频严格改善；共同real AUROC至少较score/simple更好者+.02。同分边界按候选最坏、控制最好判断。计数精确、浮点容差1e−10，非统计显著性标准。

95%全帧比较仍只按无GT风险分位取843个，全输出642好导致FA下界201，不能作为最终判断准确目标。报告90/95%固定次级点、候选实际接受数的score/simple/A0对照、共同/自身可用支持及不可用好坏。逐域/逐视频FA/FR/ED/CR、接受覆盖、正确保留/检出、全帧正确覆盖全部保留；中心仅输出帧命中率＋输出覆盖＋全帧正确中心覆盖。R与Q1均成立后才另立正确保留/误拒约束的最终工作点版本。R或Q1失败收束该版本并保留simple，不延长轮数/搜系数/反号/按视频回退或用TEST选候选。

### 22.4 本地验证及服务器运行状态

**本地完成（事实）：** 新增8个配套文件：`port_size_halfpeak_range_v3`的run/eval入口、协议/来源、NumPy监督几何、Torch监督/梯度、比较模块及21项必要测试。未修改旧v2或其清单；新清单精确pin7个直接文件、原模板清单及旧曲率清单，且调用旧来源检查，防止复用依赖悄然漂移。新老来源检查均通过。

21项测试在本地Python3.8/Torch1.8.0.post3和另一套Torch2.9.1 CPU全部通过、无跳过，包括幅值/背景不变性、GT短边/边界/padding保护、宽高交换/π/反射/等比例、两保护分支实际Torch有限差分、GT中心/背景均有梯度、低于背景仍有梯度、长短边±15%梯度方向、A0/跳过辅助时原像素梯度逐值不变、真实侧分支三项梯度与一步合成更新、实际保存重载、故意损坏共享seed/VJP必须在optimizer前停止，以及3072次更新的模拟配对调度、阶段来源/失败保护、同分界/覆盖/留出/GT接口约束和预设R/Q1判断。合成Gaussian在原未改半峰读取器恢复对应两条边，误差满足atol1e−6/rtol1e−7；这是受控解析/工程例，不是新图像精度收益。

新比较模块直接复用旧汇总语义，额外记录各域/视频参考长短边有符号P10/中位/P90、不可用原因、离线GT范围代理损失和强度保护触发帧数；这些GT代理在在线读取之后附加，不影响风险或标志。用用户原回传1319行在内存重放，既有混淆/参考/排序/工作点汇总精确一致，新增预设审核正确保留v2的R/Q1失败，没有新结果副本。Python3.8语法、CLI `--help`及`git diff --check`通过。

**未验证部分：** 本地没有原服务器冻结权重/全部运行产物，未启动实际TRAIN/VAL、服务器check、GPU smoke或新模型推理；本地真实autograd用合成图/特征，不能替代真实14视图的强度门槛、GPU保存重载或性能结果。全部新阶段终态仍需服务器确认，训练/评价的`REVIEW_REQUIRED`不表示R/Q1通过。没有连接服务器、源码打包或Git提交；通过Git同步全部新增文件后运行。其他窗口现有修改保留。


### 22.5 服务器逐条指令与单一分析回传文件

在`mmrotljj`环境，通过Git同步新增8文件并保留原父依赖、旧权重/策略/结果。环境设置合并一条；每个阶段命令单独复制执行，前一步成功后再下一步。GPU物理卡沿用3，对进程内部`--gpu 0`。新输出目录不得已存在；不删除或覆盖旧目录。v3从新初始化开始，不需旧失败帧replay、不续训旧候选。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD && export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 && set -o pipefail
```

```bash
python -m unittest discover -s tests -p 'test_port_size_halfpeak_range_v3.py' -v 2>&1 | tee work_dirs/port_size_halfpeak_range_v3_unittest.log
```

```bash
python crane_project/tools/run_port_size_halfpeak_range_v3.py --mode check --out-dir work_dirs/port_size_halfpeak_range_v3_check
```

```bash
python crane_project/tools/run_port_size_halfpeak_range_v3.py --mode smoke --gpu 0 --check-report work_dirs/port_size_halfpeak_range_v3_check/check_report.json --out-dir work_dirs/port_size_halfpeak_range_v3_smoke
```

```bash
python crane_project/tools/run_port_size_halfpeak_range_v3.py --mode train --gpu 0 --check-report work_dirs/port_size_halfpeak_range_v3_check/check_report.json --smoke-report work_dirs/port_size_halfpeak_range_v3_smoke/smoke_report.json --out-dir work_dirs/port_size_halfpeak_range_v3_train
```

```bash
python crane_project/tools/run_port_size_halfpeak_range_v3.py --mode assess --gpu 0 --train-report work_dirs/port_size_halfpeak_range_v3_train/train_report.json --out-dir work_dirs/port_size_halfpeak_range_v3_assess
```

工程放行：21项unittest无跳过；`HALFPEAK_RANGE_NUMERICAL_CHECK_PASS`；`HALFPEAK_RANGE_TRAIN_PRECHECK_SAVE_RELOAD_PASS_DISCARDED`。固定训练及评价完成为`HALFPEAK_RANGE_PAIRED_EPOCH04_COMPLETE_REVIEW_REQUIRED`、`HALFPEAK_RANGE_PAIRED_ASSESSMENT_COMPLETE_REVIEW_REQUIRED`，后续依据R/Q1与分域/视频完整结果判断。若check/smoke失败不继续train；保持原输出，不能改系数/清单/跳帧来放行。

四阶段完成后，以下一条命令在项目`work_dirs`下产生一个分析包，包含单测日志、四阶段JSON/JSONL/失败记录（若有）/checkpoint SHA标记；排除`.pth`，不需要原图或新本地结果副本。后续分析优先`initial_gradient_report.json`、`smoke_report.json`、`train_report.json/train_steps.jsonl`、`assessment.json/paired_assessment_rows.jsonl`及各completion/source合同。

```bash
tar -czf work_dirs/port_size_halfpeak_range_v3_analysis_20261005.tar.gz --exclude='*.pth' work_dirs/port_size_halfpeak_range_v3_unittest.log work_dirs/port_size_halfpeak_range_v3_check work_dirs/port_size_halfpeak_range_v3_smoke work_dirs/port_size_halfpeak_range_v3_train work_dirs/port_size_halfpeak_range_v3_assess
```

### 22.6 v3服务器回传：范围项参与训练，R/Q1仍未成立（2026-10-05）

**来源与执行状态（事实）：** 用户提供`/Users/mac/Downloads/port_size_halfpeak_range_v3_analysis_20261005.tar.gz`及终端文本；分析包SHA为`022da410a82103fd4ab83523f5d845a0c2d57da2f7d09d7438b11c6b684f26cb`。34个成员，无重复路径、越界路径、符号/硬链接或failure文件；仅在内存读取、重算，没有解包或新增本地结果副本。包内21项单测7.985秒全部通过、无跳过；数值check、丢弃式GPU smoke、固定epoch04两臂训练、1319行评价均完成。终端前一条误输入smoke命令被用户`^C`取消，随后正确命令成功；它不是训练失败。实际物理GPU为2、进程内`--gpu 0`，22.5的卡3指令保留此前下发身份。

四阶段contract一致；completion内全部文件SHA、阶段链SHA、候选checkpoint标记SHA及当前本地协议/直接源码清单精确相符，来源清单SHA为`181e6dadeb9a6e794528391f4c91a2787853e54a2e5f4563a048fe23c09c6856`。沿用第1节B24、正式midpoint23和冻结simple policy原身份。新两臂同初始化state SHA `d6b3743da3edad8901aaa1c025780141fa4272817f849e9307385246ba71886b`，384拟合帧、每臂4×384=1536次更新、共3072次，逐轮图像顺序一致；fit身份SHA `524b17705346cecd6c4cb87294018aea8aa5e2ba86cff5fd3959089b62f3e58e`。未训练参考留出/VAL、未读TEST、未选epoch。最终A0/A1 state SHA分别为`48dddf8f33ac92812cbd44d4ac18de1de19c6fd01684a28a11e976f4d95f1beb`、`03822f00010ab831371e96ed4a24821edc3c92a9437db7383647a8e5137e4179`；checkpoint字节SHA分别为`e12f669d23406df8a43594cae65005465984f1ebde60601c9e0284254283a626`、`1ce293fa155d1fc627c8ba409f176549cd4c105a7801b72797c834f2103f4165`。包未含`.pth`，本地仅核对标记与服务器保存/重载报告，不声称重算实际权重字节SHA。未对已不在Downloads的旧v2回传包重新核对A0 state；历史比较以21.11的既有核验记录为准。

评价行包括参考留出TRAIN432（real_seq13 304、sim_seq08 128）和VAL887（real_seq07 226、real_seq14 149、sim_seq10 512）。用本地已核对的比较模块重算全部1319行，参考精度、混淆、排序、工作点、范围/有符号诊断和预设继续审核均与保存报告精确一致。检测器/midpoint及候选评价前后state不变，框最大绝对差0，score/输出数量/中心与方向标志保持，GT_online=false。`REVIEW_REQUIRED`只表示完成待审核，不能读成性能通过。

**E工程门槛（事实，通过）：** 14个固定拟合视图的数值检查和真实Torch辅助有限差分通过；两臂初始28次组件测量中三项在stem/output均非零且有限。真实8次smoke更新及48次TRAIN抽测的shared-logit/same-seed VJP共56次全部通过。smoke的`sim_seq08_00000` A1旧多种子参数加和诊断有1次passed=false，但它明确为diagnostic_only，严格同上游核对通过，不能将其计成再度梯度故障。smoke和每轮保存/重载通过；没有裁剪导致的实质梯度衰减，3072步保留比例最小`.999999873`。

| 初始化A1分域 | 原像素梯度范数中位 | inner raw / ×.125 | outer raw / ×.125 | 新增合成/原项比中位 | 合成裁剪前范数中位 / 保留比例 |
|---|---:|---:|---:|---:|---:|
| real（10视图） | .152297 | .239153 / .029894 | .230761 / .028845 | .391633 | .168936 / 1.0 |
| sim（4视图） | .158939 | .921677 / .115210 | .915779 / .114472 | 1.474755 | .299592 / 1.0 |

各列是各自中位数，不能把两个分量中位数直接相加当合成范数。初始化两域通过预设[.01,10]强度范围；这只证明新增项实际参与更新，不证明`.125/.125`最优。TRAIN第4轮抽测新增/原项中位real .521256（4点）、sim 1.548220（2点），对应夹角余弦中位 .844912/.175231；样本稀疏，不能据此断言唯一根因为梯度冲突。第4轮动态训练过程内/外平均loss的均值A0→A1：real .055544→.041640（降低约25.0%），sim .066567→.043639（约34.4%）；原像素loss real .024218→.024854、sim .027234→.028115。这是逐步更新过程统计，不是固定epoch04拟合集的末态精度。

**R参考精度（事实，未通过）：** 下表误差为参考规范长短边最大相对误差的均值/P90，在A0/A1共同可用输出上计算；10%内数量同时核对各臂自身可用集合和全输出覆盖，不用剔除不可用正确帧制造收益。

| 角色/视频 | 共同输出 | 均值 A0→A1 | P90 A0→A1 | 参考10%内数 A0→A1 | 各臂可用输出 A0→A1 |
|---|---:|---:|---:|---:|---:|
| 留出TRAIN real_seq13 | 304 | .198606→.156671 | .296883→.249414 | 19→68 | 304→304 |
| 留出TRAIN sim_seq08 | 127 | .290371→.303732 | .646098→.715917 | 46→50 | 128→127 |
| VAL real合计 | 373 | .196435→.185992 | .276435→.285150 | 65→72 | 373→373 |
| VAL sim/seq10 | 510 | .169285→.191242 | .316439→.349313 | 205→175 | 512→510 |
| VAL real_seq07 | 224 | .198900→.218117 | .273255→.319069 | 48→22 | 224→224 |
| VAL real_seq14 | 149 | .192730→.137697 | .281780→.222856 | 17→50 | 149→149 |

real VAL均值仅降低5.32%，低于预设20%；P90升高，10%内全输出覆盖65/374→72/374，仅+1.87个百分点，低于预设+10个百分点。sim VAL均值恶化12.97%，10%内全输出覆盖205/512→175/512（−5.86个百分点）；real_seq07为48/225→22/225（−11.56个百分点）。留出real改善成立，但其冻结检测框仅4个尺寸错误，不能把参考留出的排序改善视作充分真实错误证据。留出sim均值/P90恶化，且不可用帧`sim_seq08_00645`是尺寸正确检测。VAL新增不可用为`sim_seq10_00344`（尺寸正确，unresolved_short_edge）、`sim_seq10_00415`（错误，not_a_gaussian_peak）；两臂共同不可用`real_seq07_00018`为错误检测/core_clipped。参考不可用不删除任何检测/中心输出。

real共同支持±15%双侧探针：VAL长边 .3646→.4558、短边 .2627→.2091；留出real长边 .5197→.6612、短边 .0559→.1283。仅留出长边满足60%门槛，不能把合成图梯度方向正确等同真实图像双侧辨识成立。

**范围代理及偏差（事实与推断区分）：** 固定epoch04离线GT范围代理的内/外平均loss在留出real .081493→.077631、留出sim .108003→.110234；VAL real .081018→.088711、sim .064189→.072460；seq07 .071589→.091378、seq14 .095320→.084666。全部1319帧代理均eligible，强度floor触发0次。因此失败不能归为本次代理被跳过或floor抹掉梯度；同时VAL的GT代理本身没有改善，不能仅称“监督已学好、只差在线读出”。TRAIN过程更低而VAL更高，提示泛化问题值得优先核对，但尚未证明输入特征、样本覆盖或读出偏差中的哪项是唯一原因。

参考/GT有符号长、短边中位：seq07 `.9449/.9001→.9341/.8506`（原偏小，尤其短边进一步偏小）；seq14 `1.0856/1.1783→1.0504/1.1175`（原偏大得到改善）；sim VAL `1.0104/1.0442→1.0394/.9725`（长边偏大、短边缩小）。这与“同一监督对不同原始偏差产生不同收益”相符；不能概括成普遍收缩两条边、也不能认为同时纠正偏大/偏小已成立。

**Q1判断（事实，未通过）：** 下表每格依次为错误接受FA／正确误拒FR／错误检出ED／正确保留CR。每行各方法接受数相同，score/simple是各自原排序的同数对照；这里分域/逐视频离线固定数量，不是把新门限分域部署。边界同分的最优/最坏FA一致，不是有利tie-break产生差异。

| VAL组 | 接受数/全帧（覆盖） | score | simple | A0 | v3 A1 |
|---|---|---|---|---|---|
| real合计 | 333/375（88.80%） | 159/12/29/174 | 158/11/30/175 | 166/19/22/167 | 161/14/27/172 |
| sim/seq10 | 510/512（99.61%） | 54/0/2/456 | 54/0/2/456 | 55/1/1/455 | 55/1/1/455 |
| real_seq07 | 196/226（86.73%） | 160/8/21/36 | 154/2/27/42 | 162/10/19/34 | 163/11/18/33 |
| real_seq14 | 137/149（91.95%） | 6/11/1/131 | 4/9/3/133 | 6/11/1/131 | 6/11/1/131 |
| pooled | 843/887（95.04%） | 216/15/28/627 | 212/11/32/631 | 217/16/27/626 | 218/17/26/625 |

共同支持error AUROC（score/simple/A0/A1）：real `.708786/.591283/.377120/.509775`；sim `.626653/.696304/.682238/.600639`；seq07 `.441162/.583207/.422727/.384596`；seq14 `.649899/.753521/.490946/.524145`。real A1虽然优于A0，但仍低于score/simple，主工作点较simple多3个FA与3个FR。逐视频均未优于较好控制；sim排序也恶化。参考seq14 10%内17→50并未带来判断优于simple，参考准确和判断准确必须分开。

**Q2工作点（事实，只作比较）：** 全局无GT的95%分位门限A0 `.987721631763944`、A1 `.9931282702186605`，各接受843个；pooled A1比simple多6个FA、6个FR、少6个ED和CR。887全帧仅642个尺寸正确输出，理论FA下界仍843−642=201。90%固定次级点接受799个时，score/simple/A0/A1为`196/39/48/603`、`194/37/50/605`、`209/52/35/590`、`212/55/32/587`，real主排序失败不是仅95%约束造成。

全局A1门限分配real359、sim484；不能直接拿其FA178/40与simple在333/510的158/54比较。按候选实际接受数匹配score/simple/A0/A1：real359为`174/1/14/185`、`174/1/14/185`、`178/5/10/181`、`178/5/10/181`；sim484为`49/21/7/435`、`46/18/10/438`、`41/13/15/443`、`40/12/16/444`。sim在这个较低接受数上有局部收益，但既有A0已经取得其大部分收益，不改变预设主510点、R和逐视频Q1均失败的结论；不能据此择点保留候选或形成最终标志。

**中心与输出（事实，冻结保持）：** VAL real输出374/375（99.73%），输出帧中心正确362/374（96.79%），全帧正确覆盖362/375（96.53%）；seq07为输出225/226、中心213/225、全帧213/226；seq14均149/149。sim均512/512。参考留出real/sim分别304/304、128/128且中心均正确。缺测帧留在全帧分母，尺寸参考不可用/拒绝不会删除可用中心。正式midpoint第18节已核验定位、sim角度和两域RIoU收益继续保留；v3失败不能改写正式midpoint没有效果。

**结论及下一步建议（尚未实施）：** 收束固定v3为“E通过，局部参考收益，R/Q1未成立”，保留基线simple和全部候选来源/负结果。当前不延长训练、搜系数/epoch、改最终工作点、反号、按视频回退或重启旧方向结构。两次辅助监督均未使真实判断超过冻结控制，下一步优先一项**不训练、固定epoch04的TRAIN读出能力对照**：在既有拟合/参考留出角色上，用同一热图比较原在线读取与离线GT中心/方向辅助的轴向半峰读取，GT长短边只用于最后评价，不作为待求边长输入；同时记录固定末态的GT范围代理，避免用动态训练loss代替拟合精度。它只用于定位当前热图可读范围与在线定位/模板求解之间的差距，不是部署候选，GT不得进入在线API。若留出下即使GT辅助也读不准，优先收束当前特征＋热图配置并另议输入/训练覆盖；若辅助读取有稳定收益而原读取没有，才有依据另立读取机制实验；这两个分支均待验证，不称已确定根因或保证以后成功。不重复完整审计、不新增实际TRAIN/VAL训练或TEST评价；具体控制与准入规则须在后续实现前冻结。

**新目录约定（用户本轮明确要求）：** 后续同一实验的所有阶段、日志、分析回传包集中于一个`work_dirs/<experiment>/`父目录，例如`check/`、`smoke/`、`train/`、`assess/`、`unittest.log`、`analysis_YYYYMMDD.tar.gz`；每阶段命令仍分别复制，必要的环境设置可合并一条。已有v2/v3目录、文件路径和来源身份保持，不在本次分析中移动或覆盖旧产物。Git同步源码，不压缩本地源码，服务器只产生该实验父目录内的一个分析包；回传仍在内存核验，不另存本地结果副本。

## 23. 冻结框尺寸残差v4：文献适配、固定对照及实现（2026-10-05）

### 23.1 问题与文献边界

**既有事实：** 22.6已表明参考精度改善不等于实际尺寸判断改善，且范围代理在VAL也未一致改善；不能把失败全部归因于读取器，也未确定特征、样本覆盖或相关误差中的唯一原因。当前midpoint TRAIN仅46个尺寸错误、sim为0；已有432帧参考留出的real仅4个错误、sim为0。连续帧数量不是独立错误类型数量。正式midpoint的定位、sim角度和两域RIoU收益保留，可靠性实验不会更换前端。

**本轮依据与推断：** 用户要求先检索论文，再明确授权按推荐实施。优先借鉴[From Keypoints to Predictive Distributions: Post-Hoc Uncertainty for YOLO-Pose Models](https://arxiv.org/html/2607.26921v1)的冻结预测、真实残差NLL监督，以及排序与分布校准分开评价的机制。它是2026预印本，原任务为关键点；原文的雾天案例也说明其分布内协方差不能稳定捕获分布偏移失败。下述OBB规范长短边、对角log尺度和小侧分支是项目适配，不能写成原论文OBB方法、PQA复现、已校准概率或已解决泛化。共享P3不保证与检测错误独立。

本版直接建模现有框的尺寸残差，取消“先准确重建第二个尺寸框”作为判断链条的前置要求。固定零均值和对角分布只是有限假设；未引入误差均值修正、Student-t、密度模型、类别平衡重采样或新辅助loss。偏大/偏小在标签与报告中保留，但NLL对同幅正负残差相同，不学习误差方向，不修正任何框。是否能提高判断准确性仍待TRAIN/VAL实测。

### 23.2 监督、可微实现与在线读取

只对冻结前端的真实输出进行离线监督，GT和预测都在**原图坐标**规范为长短边。令边长为`d_j`，`j∈{L,S}`：

\[
e_j=\log(\hat d_j/d_j^{GT}),\qquad
\sigma_j=\operatorname{softplus}(z_j)+0.001,
\]
\[
\mathcal L=\frac12\sum_{j\in\{L,S\}}
\left[\frac{e_j^2}{2\sigma_j^2}+\log\sigma_j+\frac12\log(2\pi)\right].
\]

唯一目标是此NLL，系数固定1，两轴均权；没有待调辅助系数。预测log边长的均值固定在原框，不训练均值或写回边长。两标量likelihood用Torch float64计算，侧分支float32；softplus、加法、平方、除法和log保留梯度，GT残差为无梯度常数。NLL可以为负，不能据负值判错。固定sigma下界是log尺度中的数值保护，不是由VAL选择的最优值。

尺寸正确事件仍为两轴`abs(pred/GT−1)≤.1`，对应**非对称**区间`log(.9)≤e_j≤log(1.1)`。在线仅接收两个预测`z_j`，据零均值对角Gaussian计算两轴区间质量的乘积`q_model`，风险为`1−q_model`。它是模型假设下的未校准评分，不是已验证的错误概率。报告分开记录NLL、均值偏差、两轴残差相关性、平方残差/预测方差、固定置信椭圆覆盖和10个固定`q_model`分箱；这些不会进入在线判断或拟合新门限。

在线输入为detached P3、最终midpoint框、图像变换信息和原图尺寸；不接收GT、domain、sequence或历史。固定使用已有`map_boxes/sample_local`的stride8、9×9、1.5倍上下文及padding support，不修改几何代码。对最终midpoint框采样，不用GT或更早的B框确定ROI。256通道经1×1卷积压缩到8通道后取support加权均值/标准差，共16个图像统计量，与既有3个框描述量一起进入19→2线性输出；总参数2096，无高维ROI展开、结构/方向分类器或候选重排。

### 23.3 两臂、TRAIN职责与固定预算

- **A0：** 同一小网络，16个图像统计量置零，只使用既有3个框描述量估计残差尺度。其stem梯度为0是预期控制行为。
- **A1：** 同网络、同初始state，使用16个图像统计量及同样3个框描述量。A1必须有真实图像路径梯度。
- 描述量标准化只用384个拟合TRAIN中的现有输出，不使用残差标签、留出或VAL拟合。seed1701；输出weight初始标准差`.001`、bias对应sigma约`.05`；同state不代表不同输入的初始输出必须一致。
- 沿用既有384拟合、432留出、32 guard身份：real_seq01/05/06/12各64、sim_seq08前部128用于拟合；real_seq13 304＋sim_seq08尾部128留出，中间32帧隔离。只重新命名评价角色为`residual_holdout_train`，原参考证据保留身份。此留出不隔离检测器、midpoint或simple，不能称完整系统独立确认。
- 固定4轮，epoch内顺序`RandomState(1701+epoch)`，两臂共享同一帧/同一冻结特征。Adam `lr=.001, weight_decay=0`，clip norm10，无增强、错误标签重采样、续训或epoch选择。每臂1536个计划slot；无输出或无可用图像support时两臂共同跳过，明确报slot和实际update数，禁止填GT框。每轮保存并精确重载，最后只评价epoch04。
- `check`只在拟合TRAIN做残差支持/标准化和独立NumPy有限差分。`smoke`在同一未更新初始化的固定14视图测实际Torch梯度，再做real/sim交替4次更新/臂并保存重载，全部丢弃。`train`从新初始化开始；`assess`才读取432留出和完整887 VAL图像，评估固定末态，不拟合概率校准或最终门限。TEST入口不存在；TEST已多次暴露，后续选择只用TRAIN/VAL。

### 23.4 预先固定的继续条件与工程保护

**E工程：** 新/父来源与冻结输入合同相符；NumPy公式/有限差分和真实Torch梯度相符；初始化两域A1 stem/output有非零有限梯度；记录NLL两部分的raw梯度范数、实际参数组范数及裁剪前后变化，smoke两域各臂裁剪保留中位≥.1。每步核对raw NLL解析导数与实际autograd，不再对多个参数backward种子做严格加和判断。候选forward/backward局部关闭TF32、固定cuDNN确定性并恢复基线backend标志。smoke、每轮权重保存重载精确；训练/评价检查冻结B/midpoint状态与输出，policy来源不变。失败保留`failure.json`，训练失败带epoch/slot/image/arm上下文，下一阶段拒绝消费失败产物。

**D分布拟合：** 在共同可用支持上，A1在留出TRAIN和VAL的real/sim平均log残差NLL均不高于A0。D独立报告，NLL改善不能代替正确/错误排序改善，也不能证明概率校准或分布外可靠性。

**Q1判断：** 主点仍为现有simple接受数，real333、sim510。按同接受数比较score/simple/A0/A1，报告FA/FR/ED/CR、接受覆盖、正确保留和条件正确率，边界同分同时给最优/最坏FA/FR，实际排序只按risk和image。A1保守FA比三个控制中的最好者在real至少减少5；sim及每个视频不得增加保守FA或FR，至少一个real视频有严格收益；real共同支持AUROC至少比三控制最大值高`.02`；各组A1风险可用输出必须覆盖全部原输出。标准按当前协议在训练前固定，属于实用继续条件，不是统计显著性或部署保证。

**工作点：** 90%/95%仅固定比较，报告843−642=201这一类接受数下界，以及候选实际接受数的score/simple/A0同数对照，防止跨域分配变化冒充收益。只有D/Q1都通过才可另立受正确保留/误拒约束的工作点版本；本版始终不创建最终policy，不自动部署或评价TEST。中心/方向继续使用原simple输出；尺寸评分不可用也不删除中心。中心命中仅输出分母，另报输出覆盖和全帧中心正确覆盖。宽高交换、π周期、等比例/原图恢复与现有深度接口约束保持。

### 23.5 本地实现和验证（事实）

新增7个文件：`run_port_size_residual_v4.py`、协议/来源JSON、NumPy残差/读出、Torch侧分支/梯度、比较模块及`test_port_size_residual_v4.py`；本记录为唯一文档更新。新清单pin9个直接文件及midpoint/已核验collection辅助清单；复用原collect/fit来源和冻结policy，不要求加载旧参考epoch04权重，不调用v2/v3训练或修改其文件/清单。新协议SHA `51e19badae905f084933ac5be776ed78d143cab7a3c2ded06df486b245ff8268`，来源清单SHA `ff0175c72214459be2329256e2752c95ce6a59cb88a53887a15490b1bd36d4af`。

本地Python3.8.20、NumPy1.21.3、Torch1.8.0.post3 CPU的26项测试最终全部通过（1.984秒，无跳过）。包括实际float64 autograd/有限差分，负NLL，尺度/OBB等价性，padding与GT-free输入，A0图像不变/A1图像梯度，实际裁剪、故意损坏导数必须在optimizer前停止，backend标志恢复，真实checkpoint/marker重载和禁止覆盖，来源/阶段失败保护，完整smoke＋评价序列化重放，以及实际两臂3072次**合成**更新的初始化/顺序/预算/末态检查。新/父清单检查和Python3.8 AST通过，CLI帮助参数核对通过。此为工程与逻辑证据；未在本地加载完整服务器冻结权重/缓存或执行真实图像GPU阶段，不能写成E真实TRAIN预检通过或性能改善。未连接服务器，未读取TEST、压缩本地源码或新增回传结果副本；保留并未修改几何窗口已有改动、旧实验产物和论文。

### 23.6 服务器逐条命令与一个分析包

先用Git同步本次7个新增文件和本交接。在`mmrotljj`环境逐条复制；上一条成功再执行下一条。物理GPU默认2，程序内`--gpu 0`；按实际空闲卡调整环境变量。各阶段目录不存在时才可运行，失败保留产物，不能覆盖续跑。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
```

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=2; set -o pipefail; mkdir -p work_dirs/port_size_residual_v4
```

```bash
python -m unittest discover -s tests -p 'test_port_size_residual_v4.py' -v 2>&1 | tee work_dirs/port_size_residual_v4/unittest.log
```

```bash
python crane_project/tools/run_port_size_residual_v4.py --mode check --out-dir work_dirs/port_size_residual_v4/check
```

```bash
python crane_project/tools/run_port_size_residual_v4.py --mode smoke --gpu 0 --check-report work_dirs/port_size_residual_v4/check/check_report.json --out-dir work_dirs/port_size_residual_v4/smoke
```

```bash
python crane_project/tools/run_port_size_residual_v4.py --mode train --gpu 0 --check-report work_dirs/port_size_residual_v4/check/check_report.json --smoke-report work_dirs/port_size_residual_v4/smoke/smoke_report.json --out-dir work_dirs/port_size_residual_v4/train
```

```bash
python crane_project/tools/run_port_size_residual_v4.py --mode assess --gpu 0 --train-report work_dirs/port_size_residual_v4/train/train_report.json --out-dir work_dirs/port_size_residual_v4/assess
```

四阶段完成后，只在实验父目录内生成一个分析包，含测试日志、全部阶段数值证据/来源/完成报告、训练日志和权重SHA标记，不含权重或源码：

```bash
tar --exclude='*.pth' --exclude='*.pt' --exclude='*.pkl' -czf work_dirs/port_size_residual_v4/analysis_20261005.tar.gz -C work_dirs/port_size_residual_v4 check smoke train assess unittest.log
```

### 23.7 v4服务器回传核验、结果与收束（2026-10-05）

**来源与核验范围（事实）：** 用户回传`/Users/mac/Downloads/analysis_20261005.tar.gz`，1,194,833 bytes，SHA `6116d97d4f06cb3871674a22d7036210ff935ea2b0e61b5d688c01ecac18127b`；终端附件SHA `5e9987c88bc3c6b53deb20e86c965c690ae0691b66550abebe0748c28149c2b3`。包中25个文件在内存读取，未解压或另存本地结果副本。四阶段status、合同和上游报告SHA相符；新9个直接来源、协议及父清单与当前代码匹配。完成清单中可用20个产物SHA逐一通过；按压缩约定缺少2个smoke及8个训练权重，但10个权重SHA标记与服务器完成清单一致。这里不声称本地重新核对未回传的权重字节或独立重跑GPU模型。

报告SHA：check `430a1d45c0dd041bd03e80235883abc0cd0fabd40b2971a2a77f7bb0534adaec`，smoke `cf7c5b7d88890bf9d5a9cd822f56bc6d0123445a652e072e90260bb229b9a6ee`，train `7245c6948811b6a616afeb702b1b54de391d5312846425095fc7b458ff7b02c2`，assessment `79e0ecd3dc182b139e7ad8acd9e41a3e9e2e5b6d2265457b15f45165cc210a68`，1319行评价文件 `a7925617e4c5563e5cd646429c65803b295a122a1d0cac006bc588800afe9200`。本地用当前公式从保存行重算：混淆、排序、工作点和D/Q1判断精确一致；只有24处双轴相关系数浮点末位差，最大7.7716e−16。所有行的前后最终输出一致标记、模型评估前后state相同及B/midpoint最大框差0与报告相符。原policy SHA仍为`1adac499369bade5ef1f79495260002697d2b698c80ebf5424d3d77438fae4af`。

**E工程与预算（事实）：** 服务器26项单测通过，23.220秒，无跳过。14个固定拟合视图×两臂共28次初始化实际autograd核对通过；A0 stem为0、A1两域stem非零，符合对照定义。smoke两域各臂裁剪保留均为1，4次更新/臂后保存重载一致，smoke丢弃；train初始身份与smoke一致、未复用更新权重。两臂各4轮×384=1536次实际更新，合计3072，未跳过拟合帧；只读取固定epoch04。TRAIN/留出/VAL/TEST职责未改；训练约726.36秒，评价约593.02秒，报告峰值约343.61/343.55 MiB。固定epoch04权重SHA：A0 `b400ea56da10566a6854cd7d5cfe03a67c080d49c0b77f9d93421f6e5728841f`，A1 `6d84561d5f63bafc7eee8b60f7ef8eafd1aa97ebcdeb5c75768a4acc81854945`。

训练全部raw导数核对与梯度有限；A1 real/sim stem梯度中位`.26515/.27734`，不是未连接图像路径。裁剪前总范数中位：A0 real/sim `1.04163/1.06053`，A1 `1.70391/2.01723`；裁剪保留中位均1。A0 real裁剪11/1024步、sim0/512；A1 real54/1024、sim14/512。极端拟合图`real_seq05_00364`残差`[.90344,.35462]`：A1裁剪前最大范数4556.40、最低保留`.002195`，对应裁剪后约10。工程容差和smoke放行满足，E通过；这不代表极端错误的有效更新强度充分。是否需要改变采样、优化或分布假设尚待对照，不能仅凭该帧归因于裁剪。

**实际监督支持（事实）：** 384拟合输出全部可用，只有8个尺寸错误：real_seq01/05/06/12分别`1/5/1/1`，real共8/256，sim0/128。432留出中real_seq13为4/304错误，sim0/128；两域合计428个正确。前文全量TRAIN46个错误是既有全量来源记录，不等于本次384实际见过46个错误，也不能将四轮重复帧计为新错误类型。VAL real为188/374错误、sim56/512；real_seq07为181/225，seq14为7/149。连续帧不等于独立错误片段；sim留出无错误，AUROC/AP应为空。

**固定simple接受数主比较（事实）：** 下表每格顺序为`FA/FR/ED/CR`，依次指错误接受、正确误拒、错误检出、正确保留。各行在本组内独立排序到固定接受数；域级点与视频级点的接受集合不同，不能把视频混淆直接相加代替域级点。主real四方法截点均无边界并列不确定性。

| VAL组 | 接受数/全帧；接受覆盖 | score | simple | A0 | A1 |
|---|---|---|---|---|---|
| real | 333/375；88.8000% | 159/12/29/174 | 158/11/30/175 | 157/10/31/176 | 167/20/21/166 |
| sim/seq10 | 510/512；99.6094% | 54/0/2/456 | 54/0/2/456 | 54/0/2/456 | 54/0/2/456 |
| real_seq07 | 196/226；86.7257% | 160/8/21/36 | 154/2/27/42 | 154/2/27/42 | 158/6/23/38 |
| real_seq14 | 137/149；91.9463% | 6/11/1/131 | 4/9/3/133 | 5/10/2/132 | 5/10/2/132 |

A1 real相对simple增加9个FA和9个FR，正确保留175→166，正确保留率94.0860%→89.2473%，错误检出率15.9574%→11.1702%。seq07增加4个FA/FR，seq14各增加1个；没有满足逐视频严格收益。A0 real仅少1个FA/FR，不达到预设5个FA幅度，不据本次VAL反选A0替换simple。sim主510点的FA理论下界为510−456=54，所有控制已达到该下界，相等只能说明该固定点未退化，不能证明A1新增辨识收益。

| VAL组 | error AUROC：score / simple / A0 / A1 |
|---|---|
| real | .710278 / .593428 / .616135 / .463652 |
| sim/seq10 | .630013 / .700149 / .703399 / .640429 |
| real_seq07 | .443998 / .585510 / .533903 / .458564 |
| real_seq14 | .649899 / .753521 / .714286 / .597586 |

留出real主接受257/304时，score/simple/A0/A1分别`2/45/2/255`、`3/46/1/254`、`3/46/1/254`、`2/45/2/255`；A1只与score持平。其AUROC`.738333`高于simple`.716667`和A0`.685833`，但只有4个错误且VAL不复现，不能据此保留候选。sim留出128个输出全正确，四方法128接受、CR128，不能评价错误排序。

**D分布与过度自信（事实）：** 平均NLL越小越好，负值合法；A1在四个预定域/角色上均劣于A0。

| 角色/域 | A0平均NLL | A1平均NLL | A0/A1名义95%联合椭圆实际覆盖 |
|---|---:|---:|---:|
| 留出TRAIN real | −1.958608 | −1.856000 | 92.7632% / 84.2105% |
| 留出TRAIN sim | −2.432280 | −2.071787 | 99.2188% / 69.5313% |
| VAL real | 3.286939 | 8.811954 | 36.0963% / 30.7487% |
| VAL sim | −.658746 | 2.383369 | 53.7109% / 25.7813% |

A1 VAL real平均sigma长/短轴`.02856/.02367`，sim`.01968/.01552`，比A0更窄。real残差均值`[−.05145,−.06085]`、双轴相关`.84567`；sim均值`[−.03913,−.03126]`、相关`.20573`。零均值对角假设与VAL残差存在偏差和相关性，尚不能证明它是唯一原因。real全部374个输出都被A1给到`q_model≥.98335`，平均`.998900`，实际尺寸正确仅186/374=`.497326`；sim平均`.999981`，实际`.890625`。因此不能把本版q_model解释为可信正确概率或用校准数值替代错误排序。

**覆盖与冻结中心（事实）：** VAL887帧/886输出/642尺寸正确。全局95%比较843接受，score/simple/A0/A1为`216/15/28/627`、`212/11/32/631`、`211/10/33/632`、`221/20/23/622`；FA下界仍为201。A1全局分配real331、sim512，与simple333/510不同；域级主表已固定相同接受数，不能据分配变化宣称提升。按域自身95%点，real357接受的FA为`173/172/172/178`，sim487接受为`50/46/45/49`，不支持改选95%保留A1。

中心real条件命中362/374=`96.7914%`、输出覆盖374/375=`99.7333%`、全帧中心正确覆盖362/375=`96.5333%`；seq07分别213/225=`94.6667%`、225/226=`99.5575%`、213/226=`94.2478%`；seq14与sim三项均100%。全部VAL为874/886=`98.6456%`、886/887=`99.8873%`、874/887=`98.5344%`。原框、score、输出数及三个冻结simple标志保持；未创建最终工作点、拟合概率校准或读取TEST。正式midpoint的已核验定位、sim角度及两域RIoU收益不受本次可靠性负结果否定。

**结论与下一步建议（推断/待验证）：** 收束v4为“E通过，D/Q1未通过”，保留B24＋正式midpoint23＋simple。事实支持“本次图像分支没有获得可用排序，分布更窄且在VAL明显过度自信”；不支持“只调门限即可修复”“图像路径没有梯度”“Gaussian是唯一根因”或“所有冻结特征都无效”。[本轮借鉴的残差分布论文](https://arxiv.org/html/2607.26921v1)自身也披露雾天分布偏移时协方差不能稳定捕获失败；[Ovadia等的分类基准](https://arxiv.org/abs/1906.02530)提示常规事后校准在偏移下存在局限。这些是边界依据，不能直接视为本项目已证明机制或借机选择更复杂模型。

最优先建议一项**仅针对尺寸的TRAIN支持覆盖检查**，尚未写代码或运行：复用已保存的2558条冻结midpoint TRAIN数值预测，将既有全量46个错误核对到384拟合、432留出、32 guard及其余未使用角色，按视频和时间连续片段区分偏大/偏小、长短轴共同偏差及极端残差，并报告正确与错误两类的描述量支持。仅统计既有数值，不训练、改框、找VAL/TEST门限或重复完整审计。当前回传不含全量TRAIN行，本地也没有该collection副本，故不能提前声称剩余TRAIN含有足够独立错误类型。若剩余TRAIN已有跨视频独立错误支持，才另立保留旧角色身份的拟合/留出重组对照；若只有少量相邻错误帧，优先补充独立TRAIN错误片段，再考虑新模型。VAL/TEST帧不得移入TRAIN。v4的在线训练loss日志不是固定epoch04拟合精度，若后续仍保留残差路线，应另报固定末态拟合表现以区分拟合不足与跨角色失配；目前不据此追加训练、反号排序、扩大网络、改变clip/分布或拟合最终工作点。

### 23.8 real/seq07错误接受分母、尺寸偏小与雪天解释（2026-10-06，只读补充）

**分母与限制（事实）：** 本次回答复用23.7包内VAL行，仍是B24＋正式默认midpoint23＋冻结simple，不混入几何窗口σ1.5。real raw尺寸错误188/374=50.2674%，simple接受框中的尺寸错误158/333=47.4474%，两者不是同一分母。seq07 raw错误181/225=80.4444%，simple接受中的错误154/196=78.5714%；real共158个FA中154个来自seq07，占97.4684%。seq14 raw错误只有7/149，当前FA4个，因此高错误比例集中于seq07，不能泛称两条real序列均约50%。尺寸正确依旧要求规范长短边同时相对误差≤10%，不以类别检出或中心正确替代。

在固定seq07接受196、只有44个尺寸正确输出的情况下，任何排序的FA至少196−44=152，simple实际154，距该接受数下界仅2个。这既反映原尺寸正确数少，也反映比较工作点保留大量输出；不证明simple整体足够准确，更不能保证降低接受数后仍会优先保留正确尺寸。低错误接受的最终尺寸标志需要另行验证正确保留/误拒约束；拒绝尺寸仍应保留可用中心。

**尺寸形态与原前端（事实）：** seq07当前长/短边预测与GT比值中位`.892859/.881857`，即偏小约10.7141%/11.8143%；225个输出中长边小于GT的90%有122个、短边146个、两边同时96个，偏大超过10%分别8/9个。最大双边相对误差中位13.5654%、P90 20.3127%；超过20%/30%/50%分别27/7/3个。超限不是只有几个极端离群帧，也不等于每个抓斗位置都错：154个尺寸错误接受中150个中心仍<15px。相同GT/原B输出下，B在seq07已171个尺寸超限；正式midpoint恢复9个并新增19个，末态181个。这是此视频的尺寸代价，不能否定正式midpoint已核验的定位、sim角度及两域RIoU收益。

**天气观察与因果边界（事实/推断）：** 本机查看现存VAL原图`real_seq07_00001.jpg`及`real_seq07_00113.jpg`，二者字节SHA均与v4评价行的`image_sha256`相符。可见降雪、白色覆盖和近镜头模糊雪花，支持用户指出的雪天场景；本轮未读取TEST或生成图片/结果副本。雪花干扰边缘、外观变化和图像分布偏移是合理候选解释。[检测鲁棒性基准](https://arxiv.org/abs/1907.07484)支持天气/图像退化可影响检测性能，但不能证明本项目seq07偏小由雪唯一造成。当前没有保持视角、姿态、背景等因素一致的天气对照；也未核对TRAIN是否覆盖同类降雪及尺寸偏小错误，不把seq07与seq14的差异当成雪天的因果实验。

**对后续建议的细化（待验证）：** 23.7的有限TRAIN支持覆盖检查应特别关注与seq07相似的持续偏小、双轴共同错误，以及对应图像条件是否有独立TRAIN片段覆盖。天气/外观条件的补充应保持TRAIN职责，不能直接将seq07并入TRAIN或按该视频调整门限。当前仅补充解释和现有数值，不改变阈值、模型、框、标志或后续实验状态。

## 24. 用户授权迁移至σ1.5／epoch03：simple三标志检查（2026-10-06）

### 24.1 身份与范围（事实）

用户确认最新检测版本为σ1.5／epoch03，并授权将本窗口可靠性检测前端切换到该版本，只做中心、尺寸、方向三分量判别。执行身份为**B epoch24＋冻结SigmaMidpointHead(1.5)／head_epoch_03**；头SHA为`16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7`，完整24轮TRAIN/VAL选择所得，累计2706次更新。B保持第1节原SHA。复用原冻结ROI cache manifest SHA `046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3`；实际加载显式σ1.5头，不能使用默认σ1头加载器、短拟合头或文件名“latest”。

第1/18/19节B24＋σ1／epoch23＋旧simple仍是**历史冻结比较基线**，其原权重、policy、门限、TEST结果及来源不变；第21—23节负结果保持该前端身份。新版本只把原simple规则移到已确认的新框上，重新拟合两个线性风险模型与覆盖门限。几何头更换与可靠性参数重拟合不能合称单因素可靠性创新；服务器实际结果尚未产生。

中心继续保留每个有效最终框，缺失时三个标志均false；尺寸或方向拒绝不删中心。simple只能给使用建议，不能把“中心保留”写成已经学会中心正确性判断。可靠性层不修改六个框值、score或输出数量。GT、域/视频身份、历史/未来帧均不进入在线`decide`。等比例变换、实际sx/sy还原、raw宽高/角度关联、π周期、全框fallback与深度接口沿用既有原生推理；不据此承诺深度精度改善。

### 24.2 新增代码与阶段（已实现）

- `crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py`：独立入口，`check → collect → fit → verify`，无TEST、probe、参考训练或新检测训练入口。
- `crane_project/utils/port_midpoint_sigma15_reliability_v1.py`：`Sigma15Reliability(policy, front_end).decide(final_box_original, image_size, method='simple')`；三标志字段与原simple一致，交付框字段为`final_box_original`。拒绝旧前端policy与错误B/σ/epoch/更新数/头SHA。
- 同名`_protocol.json`、`_sources.json`及`tests/test_port_midpoint_sigma15_reliability_v1.py`：固定规则、97文件来源闭包和迁移测试。旧runner/util/protocol/manifest未修改。

`check`只核对源文件、既有选择记录、selected头及B字节SHA、TRAIN/VAL数值来源、缓存metadata和旧policy SHA，不反序列化张量、不读图、不调用GPU或重建缓存。复用σ1.5入口的已审查选择核对函数；只核对已有选择，不重新选择或重复完整审计。旧policy只作离线对照，固定SHA为第1节值。

`collect`只加载`train_s1.pt`与`val_s1.pt`，冻结头读取2558个标准尺度TRAIN和887个VAL框；不加载B、不提特征、不加载半尺度TRAIN。不改原缓存。确认TRAIN/VAL B与既有simple数值来源一致，VAL重现锁定epoch03框和fallback；VAL后续用锁定文件中的原值，避免数值重放被误作新预测来源。

`fit`在新前端标准尺度TRAIN2558上标准化并拟合size/angle，规则保持三特征、两组四系数、正负各总权重0.5、L2=0.1、float64 Newton最多100次、梯度容差1e-8。方向TRAIN资格仍使用已核对real轴线／Webots OBB资格；VAL GT只离线评价，不进入参数或覆盖门限。沿用单一pooled VAL95%全帧覆盖工作点（887帧至少843个接受，边界ties整体保留），这是**固定比较工作点，不是最终准确性目标或保证**。不做门限搜索、分视频/分域路由、v2/v3/v4续训。

`verify`以冻结B24＋显式σ1.5头进行全887帧原生VAL推理，复用原GT-free `capture(detector, head, image, metas)`；既有函数/审计事件名称包含`test`不代表访问TEST，实际dataset明确仅VAL。每帧一次特征提取、三次原生B头调用，累计887／2661；核对锁定几何框、score、输出数、fallback、三个标志、正确性与四类判断计数，以及模型状态不变。框的原数值公差不改；标志或计数不同则失败并保留现场，不自动扩大容差、重拟合或调门限。

### 24.3 报告与解释边界（已实现／待测）

`fit/fit_report.json`记录TRAIN拟合内描述、VAL覆盖校准描述；`verify/verify_report.json`记录全量原生VAL检查及同一评价。均按overall、real/sim、逐视频报告：错误接受FA、正确误拒FR、错误检出ED、正确保留CR、输出条件状态准确率、接受后正确率、接受覆盖和缺失。中心另报输出覆盖、仅输出帧命中率和全帧中心正确覆盖；方向可评价GT范围与全部在线标志覆盖分别记录。

同一组**新σ1.5框**上比较raw、固定score和新simple；另外报告固定旧σ1simple直接迁移（仅离线）、匹配新simple接受数的score和旧simple排序。匹配数在各域/视频及方向资格集合内分别计算，离线按image打破ties；不当作在线门限。四状态与接受数下界一并给出，防止把95%覆盖造成的FA下界误判为纯模型问题。这些控制评价新前端内的判别差异，不替代旧基线的原TEST成绩。

继续条件：先通过来源、冻结状态与原生全量VAL一致性检查，再读取分域/逐视频FA/FR/ED/CR及同接受数对照；未形成排序/正确保留收益时不称可靠性改进。若需要最终尺寸工作点，应另外预先规定正确保留／误拒约束，在TRAIN/VAL另立版本；本轮不同时优化门限。TEST已多次暴露，本轮完全不读TEST，也不重选检测器或用TEST解释来选择参数。

### 24.4 本地验证与服务器命令

**事实：** Python3.8／Torch1.8 CPU本地17项测试通过，包含σ1.5真实头中性与空输出前向、前端/旧policy身份拒绝、原图框无损保留、width-height交换/π周期/等比例特征不变、GT-free签名、VAL GT不影响拟合门限、同数控制与方向分母、嵌套训练proof到cache身份、合成3445行的collect读取／拟合／保存重载和产物篡改拒绝。CLI帮助和`git diff --check`通过。合成数据不是项目性能证据；本机未运行服务器真实缓存、实际检测权重或GPU全VAL检查。

服务器通过Git同步新源文件。各条独立执行，每条一行；某一步失败就保留该目录和日志，不运行后续阶段。首次目录统一为`work_dirs/port_midpoint_sigma15_reliability_v1`；重跑已存在阶段时使用新的`--run-dir`，不要覆盖旧产物。所有需要的原输入沿用既有路径，若服务器归档改变路径，用对应显式参数指向原文件，不重建证据。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"; export CUDA_VISIBLE_DEVICES=3; set -o pipefail
mkdir -p work_dirs/port_midpoint_sigma15_reliability_v1
python -m unittest discover -s tests -p 'test_port_midpoint_sigma15_reliability_v1.py' -v 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1/unittest.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode check 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1/check.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode collect --gpu 0 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1/collect.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode fit 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1/fit.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode verify --gpu 0 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1/verify.log
```

GPU3通过`CUDA_VISIBLE_DEVICES=3`映射为程序`--gpu 0`；沿用原运行设备与native库身份约束。流程只重拟合两个simple线性模型，不训练B／midpoint／尺寸参考。

全部通过后，只需回传下面一个分析包；包在该实验父目录内部，包含阶段JSON/JSONL、policy、来源/身份和日志，不包含pth/pt缓存或权重，也不压缩本地Git代码：

```bash
tar -czf work_dirs/port_midpoint_sigma15_reliability_v1/analysis_20261006.tar.gz -C work_dirs/port_midpoint_sigma15_reliability_v1 check collect fit verify unittest.log check.log collect.log fit.log verify.log
```

新policy目标路径为`work_dirs/port_midpoint_sigma15_reliability_v1/fit/policy.json`，尚待服务器生成与检查；第1节旧policy保持不变。没有改动几何窗口、深度窗口、实际大小论文或新增本地服务器结果副本。

### 24.5 cachefix1：独立B缓存跨推理数值比较修复（2026-10-06）

**失败与原因（事实）：** 用户服务器首次`--mode check`在`prepare → validate_b_pair`的`real_seq07_00001`失败，尚未创建check阶段目录。原实现把旧可靠性B缓存与原生midpoint B缓存视为同一次推理产物，要求两者score逐位相等；这是本次迁移代码的错误要求。前者由常规`detector(return_loss=False, rescale=True, **batch)`推理保存，后者由冻结特征捕获后`simple_test_from_features`独立推理保存；冻结权重相同不保证两个独立缓存逐位相等。

本机只读已有`work_dirs/port_reliability_branches_v1_server_review_20261003/val_qualities.jsonl`（SHA=`d0c84a58cbb3aa589874026fdc8d0b87bbf43ef1b747bb5f504cf1c42c7359b1`）与既有σ1.5封存目录内`val_epoch_03.rows.jsonl`（SHA=`691599c82d5aa90d4bde4af2e72daa76cd53ca2eaf33003ee391ac4feb49fa34`，与其artifacts索引相符）；没有复制、重新解压、改写预测或访问TEST。首帧旧／原生B score为`.33164161443710327/.33164167404174805`，差约`5.96046e-8`。887帧中886个输出、745个score末位不同，missing和逐帧GT／domain／sequence／frame_id一致；六字段最大绝对差为`[0.0001220703125, 0.00006103515625, 0.0003204345703125, 0.0001068115234375, 0.0000011920928955, 0.0000011324882507]`，前四项单位px，第五项rad，第六项score。其中一个尺寸字段亦超过原跨缓存`atol=1e-4,rtol=1e-6`条件。差异量级符合float32数值误差，不能据此称检测器变更或性能改善。

**修复范围（已实现）：** 仅独立旧B缓存与原生midpoint B缓存的比较使用固定公差：前四坐标`abs_delta ≤ 5e-4px + 1e-6*abs(旧值)`；角度`≤2e-6rad`；score`≤2e-6`。source／权重／head／cache SHA门控、frame身份和缺失输出仍要求一致。所有数值先验证有限且符合有效OBB+score约束；超限报image、六字段绝对差与允许量。`check/check_report.json`及执行contract记录VAL比较；`collect/independent_B_cache_comparisons.json`记录TRAIN/VAL按域、逐视频最大差、对应帧和score末位不同数量。比较只读取原值，不把旧score写入新框、不折算或补框。

**严格边界不变：** 同一次原生B→midpoint推理的score和输出数仍逐位一致；可靠性层六字段无损保留仍逐位一致；锁定epoch03与collect／原生VAL重放仍沿用原`verify_replay`与原生几何验证公差，score逐位一致。没有扩大原生重放公差、改变95%门限规则或放过真实漂移。修复的数值公差只用于跨来源身份兼容，不依赖GT正确性、不改变质量模型或几何参数。

**本地验证（事实）：** 修复后20项测试通过。新增首帧真实数值回归、像素／角度／score边界与漏检／NaN拒绝，以及同一链路score仅变化`6e-8`仍拒绝的测试。只读现存完整887帧VAL交叉比较通过，原行fingerprint前后相同；97文件source闭包仍通过，所有继承源未修改。服务器真实TRAIN缓存及GPU collect／verify仍未运行，不能把本地数值回归称为服务器完整迁移成功。

通过Git同步本次四个源文件更新（runner、protocol、sources、tests）及本交接。保留首次目录与失败日志，新的各阶段和日志集中于`work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1`；依次执行，每条一行，失败则停在当前阶段：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"; export CUDA_VISIBLE_DEVICES=3; set -o pipefail; mkdir -p work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1
python -m unittest discover -s tests -p 'test_port_midpoint_sigma15_reliability_v1.py' -v 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/unittest.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode check --run-dir work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/check.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode collect --gpu 0 --run-dir work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/collect.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode fit --run-dir work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit.log
python crane_project/tools/run_port_midpoint_sigma15_reliability_v1.py --mode verify --gpu 0 --run-dir work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 2>&1 | tee work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/verify.log
```

全部完成后，仅回传一个分析包；不压缩本地代码、模型或ROI缓存：

```bash
tar -czf work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/analysis_20261006.tar.gz -C work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 check collect fit verify unittest.log check.log collect.log fit.log verify.log
```

### 24.6 σ1.5／epoch03三标志服务器结果核验（2026-10-06）

**来源与完成事实：** 用户回传`/Users/mac/Downloads/analysis_20261006.tar.gz`，包SHA=`e11e1bb8d82596082c9cbfea521b5a02a12a98a485449749b343a8ab9a32197a`，连同终端附件直接在内存读取，没有解压或新增本地结果副本。四阶段check／collect／fit／verify均完成，完成清单中的全部产物字节SHA、四份执行contract／input_check、97文件来源闭包及协议与当前代码匹配；没有failure、软硬链接、重复或越界成员。服务器20项测试通过，66.048秒。冻结前端确为B24＋σ1.5／epoch03、2706原训练更新，未新增B／midpoint／参考更新；原生887帧VAL一次特征提取／三次B头调用，共887／2661，模型state前后相同，GT_online=false、TEST未访问。

新policy SHA=`38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f`，服务器路径`work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json`。size风险门限`.41665113433410333`，angle风险门限`.4305892873822406`；score对照风险门限`.7043236196041107`（risk=1-score）。两个simple各接受843/887=95.0395%，为pooled全帧覆盖，不是每域／视频各95%。旧policy和门限不改。

**独立离线复算范围：** 收集文件为TRAIN2558＋VAL887行，VAL指纹与锁定epoch03合同一致。fit和verify全部导出框与锁定VAL六值逐位相同，三标志本地重放一致，逐组从保存标志与GT重算四类混淆／覆盖指标一致；本地NumPy风险计算仅存在float64末位差，≤1e-12，没有标志变化，不能误报同机保存重载失败。完成包不含B／head权重或原ROI张量，故此处不声称本地重新加载实际权重或重新运行GPU；服务器完成链已记录实际输入验证和冻结状态。

工程四阶段核验已完成，具体性能表、同接受数对照与分析已迁入[结果文档第6节](obb/可靠性实验主要结果与对比分析_20261006.md)。冻结旧policy／门限／权重及来源保留；本次没有训练、调参或TEST访问。

## 25. 结果文档迁移记录（2026-10-06）

用户要求结果集中放入独立结果文档，交接仅存记录。本轮撤回此处原第25节的实验主表及写作分析，完整迁入[可靠性实验主要结果与对比分析](obb/可靠性实验主要结果与对比分析_20261006.md)第1—5节；24.6的完整核验与性能表迁至该文档第6节，本处保留执行来源。该文档第7节只读引用现有Webots证据，区分共同尺寸瓶颈、可靠性覆盖／排序限制与深度公式敏感性。

## 26. 可靠性主线恢复与尺寸实验暂时封存（2026-10-07）

### 当前范围与进度（事实）

目标改为：固定最终原图框，优先改善尺寸标志对错误的识别与正确观测的保留；中心继续保留有效输出，方向按现有证据单独推进。深度数值优化与尺寸修框暂时退出当前主线，不把两项下游共同成功设为本轮验收条件。

- 当前版本：B24＋σ1.5／epoch03，头SHA `16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7`；simple policy SHA `38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f`。第24节check／collect／fit／原生VAL verify及结果文档第8节冻结TEST已完成，没有未完成的旧迁移任务。
- TRAIN2558仅拟合两组四系数线性风险；VAL887以原pooled95%全帧覆盖规则固定门限。TEST1440为已多次暴露的冻结报告，不是新选择集或未接触的独立确认。
- 当前TEST尺寸simple `FA/FR/ED/CR=275/34/67/1055`，同数score为`297/56/45/1033`；方向为`145/41/31/1214`，同数score为`157/53/19/1202`。simple有总体增量，但不是逐视频稳定优势。中心输出1431/1440、仅输出命中1430/1431、全帧中心正确覆盖1430/1440；中心不是新增错误判断器。
- 历史普通ROI／structure、Gaussian二阶矩参考、半峰读取、曲率v2、范围v3及残差分布v4均已完成相应有限验证，尚未形成一致判断优势，不进入现simple，不默认重开。v2—v4基于旧σ1／epoch23，不能直接写成当前σ1.5的候选结果；原身份见结果文档第2—3节。

### 已确认的问题与未确认原因

| 已确认限制 | 直接证据 | 对下一项的含义 |
|---|---|---|
| 输入主要是统计代理，缺少框与图像是否一致的直接核对 | simple只有score logit、相对几何尺寸、长宽比；固定两边与score而改变角度，descriptor精确不变 | 特别不能称方向评分已建立方向一致性；增加图像证据仍需真实判别收益，不能预设有效 |
| 错误排序跨视频不稳定 | 当前TEST尺寸seq03的FA59对同数score75；seq04为194对191；Sim全接受，22个尺寸错误均未检出 | pooled少22个FA不能代表逐视频一致；只调门限不等于排序增强 |
| 95%工作点与降低FA存在硬取舍 | 当前VAL seq07正确63、接受193，任意排序至少FA130，实际132；当前TEST正确1089、接受1330，下界FA241、实际275 | 应分开评价排序与使用点；下界仅由同一固定框、正确标签及接受数量定义，不是可部署GT规则或已达到的能力 |
| 风险分布和错误支持跨角色失配 | 当前TRAIN尺寸错误71/2558（Real70、Sim1），VAL Real175/374；旧v4实际fit只有8个错误、留出4个 | 全TRAIN数量不代替实际拟合曝光；类平衡不增加错误类型/场景，不据此要求新视频或认定唯一根因 |
| 参考精度/分布loss未转化为稳定判别 | 历史参考候选局部变准仍未通过判断门槛；v4 VAL过度自信且FA/FR增加 | 不以参考MAE、loss或输出概率替代正确/错误排序与四状态评价 |

当前VAL门限直接回放到TRAIN，尺寸接受1463/2558、FA13、FR1037、ED58、CR1450；其中Real接受715/1810、FR1037，Sim全接受。此为既有`fit_report.json`的**拟合内描述**，不是新TRAIN选择点、训练失败结论或泛化估计；与VAL843接受/FR10差异说明单一覆盖门限不能被当作所有角色均95%覆盖或普适判断准确性保证。不能据此在TRAIN/TEST反选新门限。

**尚未查明的原因：** 错误样本独立事件支持、共享特征信息上限、参考与检测误差是否同向、监督/标注影响和天气因果均未隔离。原框尺寸错决定可保留正确观测的上限，但修框不是本轮判断器任务；目前最核心的可操作问题是稳定的错误排序及正确保留—错误接受取舍，没有唯一已证明的训练根因。

### 下一步建议（待设计，不自动实现/训练）

先制定一项**尺寸可靠性验收合同**，而非立即选新头：保持当前前端与旧policy，复用已有TRAIN／VAL风险与正确性标签，区分①同接受数量下能否同时减少FA/FR，②同正确保留或预先规定的FR约束下能否减少FA及代价。报告raw／score／simple、原95%点及固定规则的完整排序曲线，按域/视频报告，保留缺失、参考不可用与中心覆盖；分组GT只能离线评价，不进入推理或在线路由。候选工作点职责/允许代价应在拟合前明确，不能依据暴露TEST或看完VAL后反复选最优。

若控制评分有可用排序而现点仅取舍不合适，下一项应是明确使用点职责的有限对照，不能称判断器增强；若排序本身不稳定，再设计有明确新增辨识证据的尺寸候选，并固定单因素及预算。当前不预定新结构、不重开旧参考、不把大量拒绝称改进，也不强迫中心/方向各训练一个新模型。TEST只承担冻结报告。模型、policy及阈值修改另需明确授权；本轮授权仅整理进度和封存。

小论文可继续整理已成立的检测/midpoint与可靠性基线内容；新的可靠性增强成果仍需实测，不能将“有三个标志”或前端更准写成判断机制全面改善。深度不作为当前小论文新优化成果，实际论文正文未改。

### 相近论文与可借鉴内容（2026-10-07，检索核验，未实施）

用户要求查找可靠性/安全相关研究，解释多轮修改未获稳定收益后应该借鉴什么。本次采用文献综述/验证流程；核对以下会议/作者原文及SAOD、检测校准官方代码入口。此表是来源—论点映射，不是本项目方法已采用或效果已成立的记录；未来引用位置统一为小论文相关工作“观测可靠性与选择性输出”，不同方法限制应保留。

| 来源与身份 | 原文支持的论点 | 对本项目的借鉴与边界 |
|---|---|---|
| R1：[Oksuz等，Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration，CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR_2023_paper.html) | 自感知检测需联合考察分布偏移、场景不确定性及包含定位质量的检测校准 | 最接近总体任务定位；借鉴分层评价，不照搬整图拒绝而删除可用中心，不把自感知称机械安全保证；[官方代码](https://github.com/fiveai/saod) |
| R2：[Geifman/El-Yaniv，Selective Classification for Deep Neural Networks，NeurIPS2017](https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) | 固定预测器及置信排序，通过拒绝在风险和覆盖间取舍；SGR在相应i.i.d.条件下控制选择风险 | 优先借鉴风险—覆盖/正确保留评价，不能将相邻视频帧直接当独立样本继承保证；目标不可行时可能零覆盖 |
| R3：[Geifman/El-Yaniv，SelectiveNet，ICML2019](https://proceedings.mlr.press/v97/geifman19a.html) | 端到端联合学习预测和拒绝以优化指定覆盖区域 | 是任务目标参考，不是当前冻结前端上的直接控制；本轮不联合重训检测器 |
| R4：[Kuzucu等，On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines，ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf) | 检测校准评价/工作门限存在陷阱；正确设计的Platt/Isotonic后处理可成为强校准基线 | 可借鉴低成本校准控制及同一使用集合上的联合评价；严格单调变换不改变排序，同数FA不能因此自动改善；Isotonic可合并ties；[官方代码](https://github.com/fiveai/detection_calibration) |
| R5：[Jiang等，Acquisition of Localization Confidence for Accurate Object Detection（IoU-Net），ECCV2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.php) | 类别置信与定位质量不等价，可另学定位质量 | 借鉴直接监督最终框分量质量，而非必须先重建参考框；IoU不等于双边≤10%，旧ROI质量路线已有失败，必须明确实质增量才另立候选，不默认重启 |
| R6：[Li等，Generalized Focal Loss V2，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.pdf) | 用边界回归分布统计构建定位质量预测器DGQP | 提示中间分布可能提供质量证据，但代理Gaussian中点热图不等于真实误差分布；不重新开放失败尺寸桶头、不从低entropy预设可靠 |
| R7：[Andeol等，Confident Object Detection via Conformal Prediction and Conformal Risk Control: an Application to Railway Signaling，COPA2023/PMLR204](https://proceedings.mlr.press/v204/andeol23a.html) | 以铁路信号为应用，通过预测集合/风险校准处理检测不确定性 | 应用背景接近安全相关感知，但全文第4.2节明确不控制false positive；其扩框/真值覆盖主要针对漏检，不能作为当前错误尺寸接受的解决方案 |
| R8：[Angelopoulos等，Conformal Risk Control，ICLR2024](https://research.google/pubs/conformal-risk-control/) | 在交换性及有界单调损失等条件下控制期望风险 | 理论边界参考；接受后错误比例不一般满足该单调损失合同，不能直接套阈值宣称FA风险保证；期望控制不等于每视频/每帧保证；[核验全文](https://arxiv.org/pdf/2208.02814) |

**概念与推断。** 当前三标志是视觉观测能否使用的判据，具有安全相关应用动机；它既不是机械危险状态标签，也没有故障率、系统级危害/控制闭环证据。多轮失败不能从文献反推出唯一根因。研究启示是分开“连续误差/参考重建”“正确错误排序”“分数概率校准”“工作点/风险控制”：后三者也不能互相替代。IoU/预测集合覆盖与本项目中心<15px、双边≤10%、角度≤3°合同不相同。

**当前采用顺序（建议，未实现）。** 先采用R1/R2的任务与评价思想：固定M及原policy，利用既有TRAIN/VAL分量标签报告`接受后错误比例=FA/(FA+CR)`、全帧接受覆盖、`正确保留率=CR/(CR+FR)`与CR/N，空接受风险记不可评价而不是0；检验score/simple是否存在可用的低风险且保留正确观测的区域，不另跑检测或尺寸训练。校准误差仅作补充，不能替代同接受数/同正确保留下的FA/FR。若有可用排序，再另立合法校准/使用点对照；若没有，则单调校准不足以解决，只有明确新增分量辨识证据和有效真实错误监督的候选才值得实施。R5/R6只是条件性候选依据，当前不选新头；R7/R8不能在事件独立性和分布合同未成立时宣称正式安全/风险保证。TEST不参与上述选择，尺寸方向平衡保持封存。

本次只进行文献核验与执行交接补充，不修改模型、policy、结果文件或论文正文，不运行训练/推理。

### 封存与本轮检查

尺寸方向平衡版本保留本机Git `9042033`的源码、协议、曝光表、来源闭包和15项本地测试，以及本地check／prepare；尚无真实GPU smoke／finite／VAL成绩。原失败连续尺寸C及其下游负结果、权重、缓存完整保留。封存只改变推进状态，不删除/移动证据或修改算法；历史命令保留在[尺寸统一运行说明](detection/尺寸诊断与有限对照_v1_运行说明_20261006.md)，恢复须用户明确授权。本轮未核验远端同步或进程，不声称已经停止远程作业。

本轮只读核对当前源码、policy字节SHA、既有fit与TEST报告；从1440帧保存的错误量与冻结标志复算上述四状态和接受数下界，与原报告一致，并验证角度变化不改变三特征。没有重新推理、拟合、选择门限或生成结果副本。仅更新本交接、尺寸运行说明及旧OBB文档身份提示；旧σ1／epoch23、当前σ1.5模型/policy/数值与论文均未修改。

未改变既有实验结论、原policy／权重／门限／结果身份，未新增回传结果副本；未修改算法、几何或深度窗口、论文，未启动训练／推理、连接服务器或访问TEST。后续新增主表／分析更新结果文档，过程和决定继续记入本交接。

## 27. 固定M与simple的正确保留—接受风险诊断（2026-10-08）

**范围与预先合同。** 用户授权实现、Git提交/推送、服务器拉取CPU运行、压缩回传、分析记录后删除本次压缩包；已明确主目标为尺寸正确保留率≥95%，另完整报告90%和99%。这是现评分的取舍/排序容量诊断，不是模型性能改进实验；不生成新policy，不选择可部署门限，不访问TEST，不恢复封存的尺寸头或深度路线。

**实现。** `crane_project/tools/analyze_port_reliability_tradeoff_v1.py`绑定原cachefix1的3445行最终M预测、fit/policy、保存VAL标志及完成回执的精确SHA；复用已有NumPy排序/混淆工具。whole-tie完整曲线含空接受点；按`ceil(target×原本正确输出数)`找首个满足点。GT只描述排序容量，诊断点不得直接在线使用。pooled约束和各域/各视频同时满足约束分别使用一个全局点，不按视频部署不同门限。固定policy、same-count score、方向资格及全帧在线标志分母分别报告。中心只用输出统计命中，另报输出覆盖及全帧正确覆盖。固定与诊断点的不可用连续段包含漏检，采样帧号间隔断开，不宣称实际停用时长。

**代码与检查状态。** 新增utility、runner、protocol/source manifest、11项针对性数值/身份测试及单一运行脚本；旧模型/policy/评分/原实验源码不变。本地系统Python缺NumPy，改用已有Codex依赖Python，无安装；11项通过、Python3.8 AST与bash语法通过。提交前还核对固定输入完整路径、SHA及原VAL回放。服务器已有未提交/已暂存历史改动，保留其worktree/index摘要；同步只作快进，不stash/reset/清理。

**服务器入口（CPU，无需显卡）。** 在项目根目录、既有mmrotljj环境运行：

```bash
bash tools/run_port_reliability_tradeoff_v1.sh \
  work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 \
  20261008_retention95_v2
```

所有本轮运行产物统一在`work_dirs/port_reliability_tradeoff_v1/<run_id>/`，包含测试、运行日志、report、完整压缩曲线及完成回执；压缩包位于`work_dirs`根目录。源码/协议/运行说明通过Git，数值输入复用服务器原产物，不把权重打包。脚本拒绝覆盖并比较运行前后暂存/工作区diff摘要。

**回传核验修复。** v1已运行并回传；本地与服务器NumPy/BLAS浮点末位差（边界差约1.11e−16）使少数非95%主结果的阈值边界帧重算计数不同，不能当作模型变化。v2只补存服务器精确逐帧评分/框/标志用于完整曲线复核，合同、输入、policy和诊断规则均不变；保留v1身份，不调整容差改变结果。

**当前待办。** v2源码同步→CPU运行回传→精确评分下独立核验与有用结论写入结果文档→删除仅本次v1/v2新压缩包。原输入archive、服务器原结果及本地分析文件保留。完成后用结果和实际状态替换本段待办。
