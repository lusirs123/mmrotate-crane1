# SymEOOD 分量可靠性交接：保留midpoint23比较基线；σ1.5／epoch03三标志迁移

**2026-10-09最新：第37节固定特征来源对照已完成GPU2完整TRAIN/VAL、两端逐帧与独立诊断复核。同836接受native FA183、midpoint189、simple/score188，有局部增量；Sim及seq14保护未通过，不采用、不读TEST、保留正式M/simple/policy。第36节冻结TEST结果与身份保持。**

**2026-10-08最后实测：第35节温度／有符号修正的有限对照已完成并独立复核，VAL_FAILED_STOP，不采用、不进入TEST；温度控制精确重现第34节，解除公式约束未解决泛化，正式M/simple保持。**

**2026-10-08当前入口：用户授权第33节“冻结VAL补评”，完整评价887帧VAL，复用第32节模型/特征和历史fit门限，不重新拟合/重定点。第32节probe失败与旧合同仍保留；补评不能追溯改为原合同通过。本地/服务器9项新增测试及887帧独立复核通过；同CR663下对边FA221、simple208，完整VAL未建立稳定增量，结束本项对边方案。正式M/simple/policy不变，未读TEST，结果与清理见33.3。**

**后续可靠性候选统一协议（用户明确，2026-10-08）：仅使用现有TRAIN／VAL／TEST，不再为每个候选新增fit/probe/legacy/purged等数据角色。完整TRAIN拟合候选及公平控制、标准化；完整VAL分域/逐视频评价同接受数/同正确保留下的FA/FR/ED/CR、覆盖与连续FR，并按预定要求确定使用门限；候选及门限冻结后TEST报告，已多次暴露TEST不调参/重选。固定中心、方向、框、score和输出/漏检身份。历史实验按其原角色和预算解释，不合并重算成新协议。当前这一次只做冻结VAL补评，不改历史训练集或工作门限。**

**2026-10-06执行入口：用户已授权将可靠性前端迁移至B24＋midpoint σ1.5／head_epoch03，继续simple三分量判断。先读第24节，最新服务器结果见24.6，修复与命令保留24.5。cachefix1的check／collect／fit／全887帧原生VAL verify均已完成，服务器20项测试通过；新simple尺寸／方向各接受843帧，Real尺寸FA144、FR10，Sim尺寸FA46、FR0。分域同接受数尺寸与score相同，seq07局部较score少5个FA；总体迁移收益含检测几何变更，尚不能称可靠性机制创新。σ1／epoch23及下述已完成TEST成绩仍保留原比较基线身份，不能写成新前端成绩。本次没有检测／midpoint／图像参考训练或TEST访问。**

**结果文档入口：[可靠性实验主要结果与对比分析](obb/可靠性实验主要结果与对比分析_20261006.md)。实验主表、前端配对比较及与深度问题的分析集中在那里；本交接保留执行、来源、修复和决定记录。**

更新：2026-10-05；用户明确将**B epoch24＋正式midpoint epoch23＋现有simple三标志**的已完成固定TEST流程作为后续可靠性优化的比较基线（第19节）。基线身份已固定，不以性能最好或先优于score作为采用前提；现有取舍是比较起点。第17节为CPU TEST入口，第18节为已核验结果，第19节区分几何变化与可靠性判别改进。第14节保留迁移事实，第15/16节保留模板候选验证，模板未进入基线。旧B＋simple TEST保留第3节历史身份，不能替代当前主基线成绩。

本文是当前可靠性基线与后续实验的唯一执行交接；机制、已记录主结果与旧基线流程集中在[可靠性统一报告](obb/可靠性实验主要结果与对比分析_20261006.md)，旧σ1／epoch23流程见其[历史第12节](obb/可靠性实验主要结果与对比分析_20261006.md#history-sigma1)，[文献与项目适配](obb/可靠性判断近期论文与项目适配_20261008.md)保持独立。原模型、policy、门限与实验结果不因文档整理改变；更早流程仍见[替换前归档](archive/20261003_reliability_replaced_by_simple_v1/README.md)。检测几何优化维护[独立几何交接](geometry_precision_handoff_20261001.md)，后续服务器结果不复制为一次性本地复核目录。

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

已更新本文、[分量可靠性主文档](obb/可靠性实验主要结果与对比分析_20261006.md#history-sigma1)、[小论文完整流程](obb/可靠性实验主要结果与对比分析_20261006.md#history-sigma1)及[文档入口](README.md)。旧模型链、旧数据结果及已关闭实验不再放入当前正文，未把历史数据改名成当前成绩。最新服务器结果为本页第3～5节唯一主结果；TRAIN/VAL仍保留其拟合/开发职责。

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

**本轮已执行的服务器入口（CPU，无需显卡；无需重跑）。** 项目根目录、既有mmrotljj环境；脚本拒绝覆盖已有ID：

```bash
bash tools/run_port_reliability_tradeoff_v1.sh \
  work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 \
  20261008_retention95_v2
```

所有本轮运行产物统一在`work_dirs/port_reliability_tradeoff_v1/<run_id>/`，包含测试、运行日志、report、完整压缩曲线及完成回执；压缩包位于`work_dirs`根目录。源码/协议/运行说明通过Git，数值输入复用服务器原产物，不把权重打包。脚本拒绝覆盖并比较运行前后暂存/工作区diff摘要。

**回传核验修复。** v1已运行并回传；本地与服务器NumPy/BLAS浮点末位差（边界差约1.11e−16）使少数非95%主结果的阈值边界帧重算计数不同，不能当作模型变化。v2只补存服务器精确逐帧评分/框/标志用于完整曲线复核，合同、输入、policy和诊断规则均不变；保留v1身份，不调整容差改变结果。

**已完成运行与核验（事实）。** 实现提交`0218d231bcf448a43113dc7daa5a606c5ae49c9f`，精确评分补充提交`7f8933e5bd8d3e112a39c52a92ee4f565f965d27`，均已推送origin/main并在服务器快进拉取。v1/v2两次CPU运行成功，各11项测试通过，v2为11项/0.120s；v1/v2全部数值报告精确一致。补评分不更改任何排序、门限、角色、输入或规则。服务器M/policy/预测输入SHA全部一致，旧暂存/工作区改动完整保留（摘要worktree `b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1`、index `d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e`）。

**回传身份。** 结果统一在两端`work_dirs/port_reliability_tradeoff_v1/20261008_retention95_v2/`，v1仍在同一父目录。v1包SHA `83a6a92d943a482ce76e133f570bbee41e9e6f52f349f15f02379657db5e7633`；v2包SHA `22798dfc47aa50633d554379996b197aec876b5140cd6416c2d6fd3f48b8da08`。v2报告SHA `4bcd7f64a71822f9a74c25d901e6c695989fdfd5f49acf920551de17e842ecbb`、完整曲线SHA `4ad847ccb4869b39b6a42ce0be5f111a9f274de21e90ffc10c233d3cb655e8fd`、精确评分SHA `02c2bf6a5ce7cd093acd013e22948251e9163eef6506dcf1a19b9a69a4bb987f`、completion SHA `3eafd63ff45a8b2c9d2fdb7c4e258f1181b1cd774f0fcdaec3612724d47818ea`。原输入`/Users/mac/Downloads/analysis_20261006.tar.gz`只读保留。

**独立复核。** 包SHA/安全路径/完成回执/artifact及104份来源逐项通过。所有3445行元数据、GT、原图框与score和原输入一致；VAL887三方法框/标志/风险精确回放。服务器精确评分下独立重算78组55335个完整曲线点、360组约束混淆数和同数score全一致；独立成对AUROC最大差2.22e−16，本地/服务器评分最大末位差2.22e−16，均已记录于`local_review.json`。不存在新模型/评分/policy训练、检测推理或TEST访问。中心TRAIN2557/2558、VAL874/886命中；全帧正确覆盖分别2557/2558、874/887。

**结果与决定。** 结果文档第9节已写入主要表格/限制：pooled95%不能替代逐视频95%，simple在Real的AUROC0.5448低于score0.6648；各域/视频95%点FA200，对同数score199无优势。seq07自身95%容量点仍接受风险68.42%。不部署诊断点、不仅靠门限/单调校准声称判别提升；后续候选先明确尺寸错误辨识证据及95%正确保留/同数FA/逐视频验收，未自动指定、恢复或训练新头。

**压缩清理已完成。** 先完成分析并写入本交接和结果文档，再逐个核对包SHA、删除仅本次`work_dirs/port_reliability_tradeoff_v1_20261008_retention95_v1.tar.gz`及`...v2.tar.gz`的本地与服务器副本，并验证均不存在。两端v1/v2原结果目录、解压后报告/曲线/评分、local_review及旧输入`/Users/mac/Downloads/analysis_20261006.tar.gz`保留；没有清理其他结果或权重。没有待运行/待回传的本轮任务，结果记录纳入Git；后续评分候选仍待单独明确，不自动启动。

## 28. 尺寸评分两项特征消融（2026-10-08）

**授权与固定合同（实现前确定，执行已完成）。** 用户授权按上一轮同样流程完成本地代码/检查、Git提交推送、服务器拉取CPU运行、条件性TEST、压缩回传分析记录，再删除本次压缩包。只检验两个候选：`drop_relative_size`＝score logit＋log长宽比，`drop_aspect`＝score logit＋log相对尺寸。固定M、相同TRAIN2558输出及真实双边≤10%标签；从零重新拟合两个3系数（含bias）尺寸线性模型。标准化仅用TRAIN，类总权重各0.5、L2=0.1、Newton≤100、梯度容差1e−8，与原simple一致；不是给旧系数置零，不另搜组合/正则/损失/预算，不训练图像头。旧尺寸与深度路线继续封存。

**VAL标定与公平控制。** 原simple评分/参数、原score及历史policy原样保留。4种尺寸评分各按同一固定规则，仅一次从VAL887及GT标定新全局尺寸门限：各层正确输出至少保留`ceil(0.95×good)`，取all/domain/video各自首个whole-tie点的最大值；在线不使用GT、域或视频路由。新工作点属于受监督VAL标定，不能把上一轮诊断点直接部署、冒充历史95%覆盖规则，或当作独立确认。候选方向/中心标志与方向风险逐帧复制原simple，原框/score/漏检身份精确保持。

**预定验收与停止条件。** 各域/视频正确保留≥95%；相同接受数下，总体FA严格少于原full_simple及score两种排序，域/视频FA均不增加。score边界等风险若被截断，同时报GT-free图像顺序和FA上下界，验收采用保守最小FA界，不利用GT选择边界帧。各统计层最长不可用观测连续段不超过两种同接受数控制，允许新增0帧；漏检计入、帧号间隔断开。若只胜simple而未胜score，至多说明减少特征负作用，不通过新判别优势验收。通过者按VAL全局FA最少、再CR最多选择；完全并列保留原方案，两项都不通过立即收束本线性配置，不调整门限/结构/权重、不访问TEST。不能据有限失败声称三特征完全没有信息。

**TEST边界。** 先保存候选参数、VAL标定、gate及选中policy的完成回执/字节SHA，独立test阶段先核验冻结选择，通过后才读取原M TEST1440及原metadata。复用框，只重新评分，报告逐域/视频、三分量四状态、输出/中心正确覆盖、联合OBB与连续性；不据TEST反选或改门限。已暴露TEST只作冻结报告，不保证其正确保留也达到VAL95%。不存在GPU/检测/几何/深度训练或推理。

**代码与检查。** 新增`crane_project/utils/port_reliability_feature_ablation_v1.py`、`crane_project/tools/run_port_reliability_feature_ablation_v1.py`、对应protocol/sources、17项定向测试及唯一脚本。147份来源封存，旧文件/manifest不改。测试覆盖旧求解器一致、删除特征不影响拟合/读出、重新拟合而非置零、单全局门限/完整tie、保护项、严格gate可通过/可拒绝、并列停止及未选中候选时不触碰TEST路径。系统Python缺NumPy，使用现有Codex依赖Python，不安装包；Python3.8语法与bash语法检查。服务器现有暂存/工作区改动保持第27节摘要，快进同步，不stash/reset/删除。

**服务器单一入口（CPU，已执行）。** 源码/命令留在Git；两项fit及可选TEST统一一个实验父目录，压缩包在`work_dirs`根目录。以下固定ID已有结果，脚本会拒绝覆盖，不要重复运行。

```bash
bash tools/run_port_reliability_feature_ablation_v1.sh \
  work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1 \
  20261008_size_features_v1
```

产物：`work_dirs/port_reliability_feature_ablation_v1/20261008_size_features_v1/`。VAL没有选中时保存`TEST_SKIPPED.txt`，不启动TEST；脚本拒绝覆盖。回传包括精确逐帧评分、candidate模型/标定、报告、源证明与完成回执，不含权重或原缓存。保留原policy与所有原结果。

**执行与来源（事实）。** 源码提交`a017764d29ebb15a9b1e3fc0e74e7a6d836aa10c`已推送并在服务器快进拉取；服务器17项测试通过，两个CPU Newton拟合分别5/4轮收敛，无检测/几何/图像训练或推理。Git首次使用失效`127.0.0.1:17890`代理失败，本次拉取最终通过逐命令清空代理完成，没有修改持久配置。服务器原暂存和工作区改动的前后SHA分别保持`d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e`、`b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1`。

**VAL实际结果（事实）。** 887帧、886输出，663个尺寸正确、223个错误、1个漏检。下表4种评分各自按同一“所有统计层正确保留≥95%”规则产生单一VAL门限；接受数不同，不能直接按FA大小判定排序优势。FA/FR/ED/CR依次是错误接受/正确误拒/错误检出/正确保留。

| 尺寸评分 | 接受数 | FA / FR / ED / CR | 总体正确保留 | 接受风险 | 全帧正确覆盖 |
|---|---:|---|---:|---:|---:|
| 冻结full_simple＋本轮VAL标定点（控制） | 856 | 200 / 7 / 23 / 656 | 98.94% | 23.36% | 73.96% |
| score＋本轮VAL标定点（控制） | 861 | 201 / 3 / 22 / 660 | 99.55% | 23.34% | 74.41% |
| 去掉相对尺寸 | 848 | 193 / 8 / 30 / 655 | 98.79% | 22.76% | 73.84% |
| 去掉长宽比 | 854 | 214 / 23 / 9 / 640 | 96.53% | 25.06% | 72.15% |

原线上policy仍是`38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f`，VAL尺寸`190/10/33/653`、接受843，不改变为表中任何标定点。新候选门限分别`.43172604349724175`、`.6036068436394765`，控制新点分别`.43131551898452714`、`.7688139975070953`；均只作本轮候选/控制产物，没有部署。

| 同接受数比较（每行保持候选的该层接受数） | 候选FA | full_simple FA | score FA | 候选最长不可用段 | full_simple / score最长段 |
|---|---:|---:|---:|---:|---|
| 去掉相对尺寸，all 848 | 193 | 195 | 196 | 5 | 5 / 4 |
| 去掉相对尺寸，Real 339 | 148 | 150 | 148 | 5 | 5 / 4 |
| 去掉相对尺寸，Sim 509 | 45 | 45 | 45 | 3 | 3 / 1 |
| 去掉相对尺寸，seq07 196 | 135 | 135 | 140 | 5 | 5 / 4 |
| 去掉相对尺寸，seq14 143 | 13 | 13 | 13 | 1 | 1 / 1 |
| 去掉长宽比，all 854 | 214 | 200 | 198 | 2 | 5 / 4 |
| 去掉长宽比，Sim 485 | 44 | 34 | 39 | 2 | 5 / 3 |

**为何没有采用（事实与解释）。** 去掉相对尺寸满足全部正确保留和同数FA门槛，5项失败均为相对score的连续不可用段保护（all/Real/Sim/seq07/seq10有重复统计层，不是5个独立事件）。Real seq07最长拒绝段为帧136—140、5帧；Sim seq10为211—213、3帧，两段全部尺寸错误、没有正确误拒或漏检。因此它不是“没有排序收益”或“连续拒绝损害了正确观测”的证据；事实是新增的尺寸不可用段超过预先规定的允许新增0帧约束。拒绝连续错误与保留连续可用观测之间的职责需区分，不能在看到这轮结果后直接放宽合同、补TEST或采用候选。去掉长宽比则有8项总体/Sim同数FA失败，总体比两控制多14/16个FA，Sim多10/5个。

**排序范围（事实／推断）。** 去掉相对尺寸 pooled错误AUROC`.7873`，原full_simple`.7856`、score`.6844`；Real为`.5405`，仍低于score`.6648`，seq14`.3586`也低于full_simple`.4152`。在相同各视频接受数下，它相对full_simple的FA均持平，pooled小幅优势包含跨视频接受分配改变，不能称视频内错误辨识普遍增强。去掉长宽比 pooled AUROC`.5474`、Sim`.5234`，不支持删除该特征。结果排除了“两个尺寸统计特征都应移除”的简单解释，但没有确定唯一训练根因。

**保护与本地独立核验（事实）。** 147份源码、6个原输入SHA、3个fit产物字节SHA一致；精确核验3445行原图框/score/GT/身份和887行原policy决定，60组混淆/排序及120组同数控制计数一致。两项最终损失与梯度独立复算通过，最大梯度约`5.0e−16`；本地风险末位差≤`2.23e−16`，AUROC差≤`1.12e−16`，门限成员使用精确服务器评分。两候选中心/方向标志及方向风险逐帧不变：VAL中心命中仅输出帧`874/886=98.65%`，输出覆盖`886/887=99.89%`，全帧中心正确覆盖`874/887=98.53%`。联合合格OBB正确数原policy/去掉相对尺寸/去掉长宽比分别621/621/598，不能将参考误差或新标定点称为几何改善。

**决定与待办。** `selected_arm=null`，两项不采用，不读取TEST；停止本轮三特征线性配置的进一步组合、系数、正则或门限搜索，原M/policy保持。去掉相对尺寸作为“有限排序收益但未通过完整合同”的离线候选封存，不能与去掉长宽比明显退化混为一谈。下一项若继续，先明确“正确观测连续保留”与“连续错误拒绝”的评价职责，再预先制定有限辨识证据合同；这不是自动放行本候选或恢复旧图像头的授权。

回传包SHA `6990ea3fa306388ef45dc6f29b2b22d87cfa1c5e8cb3ac9f46a43875a0d86a80`。两端原结果集中于`work_dirs/port_reliability_feature_ablation_v1/20261008_size_features_v1/`，本机`local_review.json`保存独立复核，服务器`TEST_SKIPPED.txt`及回执确认未进入TEST。先完成分析并写入记录，再核对SHA删除仅本次本地/服务器压缩包，已验证两端均不存在；解压结果、候选参数、日志、旧输入包和所有历史结果保留。本轮执行/回传/记录/清理均已完成，无待运行任务。

## 29. 连续状态分开统计及未来验收职责（2026-10-08）

**授权与范围。** 用户授权实现上一轮建议、必要时在工作目录连接服务器。本机既有证据足够，本轮只在`/Users/mac/Documents/paper/symEOOD`操作，不连接服务器、不提交/推送、不重新训练/推理/拟合、不读TEST。保留第28节源码/合同/结果/policy及未采用状态；结果文档已有用户修改原样保留，在第11节追加补评。

**实现。** 新独立utility `crane_project/utils/port_reliability_state_continuity_v1.py`、runner `crane_project/tools/review_port_reliability_state_continuity_v1.py`、protocol及17项测试，不修改任何旧评分或gate。输入按completion内SHA绑定三个fit产物与回执；原图框/score、中心/方向标志、候选门限决定逐帧核对。补评FA/FR/ED/CR/MISSING及各类连续段、不可用段构成、切换；missing不参与混淆。使用reliability_role划分TRAIN2558/VAL887，原sim的train_sim身份不改。源码/输入SHA写入报告，已有输出拒绝覆盖，输入/输出必须位于项目目录。

**执行与检查。** 本地17项测试通过（无NumPy/Torch依赖，无skip）；覆盖连续拒错不算误拒、混合不可用组成、漏检不算ED/FR、帧号间隔/视频边界、规范长短边10%边界、ties的GT-free排序、禁止TEST/源角色冲突、篡改回执及覆盖拒绝。初版回放发现源sim split为train_sim，已改按原reliability_role读取并添加测试；该次无结果目录写出。最终在同一父目录`state_continuity_review_v1_final/`完成3445行补评、历史VAL混淆/连续性/同数控制回放、输入/源码哈希及Python3.8语法检查。`state_continuity_review_v1/`为开发过程中早期只读补评，最终报告/receipt/local_review以final目录为准。未生成回传压缩或本地源码副本。

**结果与决定。** 详表集中结果文档第11节：去掉相对尺寸同848接受FA193/FR8，对完整simple195/10及score196/11；最长FR1对1/2；ED5对5/4。seq07的136—140和Sim的211—213全部ED。候选FA最长12对完整simple11，Sim为8对score7；故正确拒错不代表稳定判别增强。历史selected_arm仍null，不以新定义重选/部署/补TEST。新合同只用于未来独立版本：95%逐域/视频正确保留、同数FA优势及分组保护、连续FR保护；FA/ED/漏检/总不可用分别报告，不把总不可用非增加当作独立否决项。GT始终只离线评价，中心覆盖保持；尺寸/深度路线封存。

**本机复现（本轮已完成，无需重跑；新输出名才可运行）。**

```bash
cd /Users/mac/Documents/paper/symEOOD
python3 -m unittest tests.test_port_reliability_state_continuity_v1 -v
python3 crane_project/tools/review_port_reliability_state_continuity_v1.py --out work_dirs/port_reliability_feature_ablation_v1/20261008_size_features_v1/state_continuity_review_v1_new
```

无远程运行/同步任务被本轮启动或停止；旧包清理状态不变，不删除任何历史文件。后续仅待决定新增辨识证据的设计，没有遗留必跑训练或TEST。

## 30. 跨尺度尺寸一致性有限可靠性实验（2026-10-08）

**授权与固定合同。** 用户授权本地实现、检查、Git同步，服务器在`/media/omnisky/personal_files/ljj/symEOOD`执行，回传分析并写入本记录后删除仅本次两端压缩包。主线为尺寸可靠性，不修改M标准输出、中心、方向、score、输出数量、原policy或深度公式，不运行封存尺寸头。以下合同在本次拟合/结果计算之前固定，源码和protocol单独SHA封存。

**机制与对照。** 标准和固定半尺度M均已还原至原图坐标，规范排序长短边；新增`max(|log L1-log L0.5|,|log S1-log S0.5|)`及辅助漏检标记，后者编码为“差异0、missing1”，不会直接删除主输出/中心。保留原simple三特征，5维线性候选对比相同样本/零初始化/类平衡/标准化/L2=0.1/Newton最多100轮、梯度1e−8的3维重拟合控制；另保留冻结full_simple和score。无新网络/参考/扰动、扫描、尺寸修框或GT在线入口。

**输入与职责。** 复用已有2558对TRAIN标准/半尺度M，只读旧有限尺寸结果的`midpoint`，不采用失败的`candidate`等字段。标准2558帧与可靠性原预测逐值一致。半尺度由RResize后固定PortIsotropicShrink生成并用原native sx/sy还原，不能再次除以0.5。沿用已封存身份划分：fit1942（Real1377/Sim565）、probe492（351/141）、legacy64、purged60；fit只按主输出存在筛选，不按正确性或中心误差排除。两尺度已参与M训练，probe也曾参与前端训练；不称独立泛化。混合TRAIN/VAL源gzip字节与角色字段需要读取，但VAL记录丢弃，不参与本阶段评分/拟合/指标；不读TEST。

**预先继续条件。** probe使用一个离线全局工作点，保护各域/视频正确保留≥95%；这不是线上门限或独立验证。总体和Real同数FA严格少于3个控制，其余分组FA不增加，控制ties取最少FA边界进行保守验收；最长FR不增加。分域/逐视频报告FA/FR/ED/CR/MISSING、接受及全帧正确覆盖、AUROC/AP、同接受数及同CR控制、各类连续段；同时报告三分量及中心的输出帧命中、输出覆盖、全帧正确覆盖。fit/legacy/purged只作诊断，不选择权重。失败即收束本版本，不追加尺度/系数/训练，不进行辅助VAL推理或TEST。通过后才另行采集固定半尺度完整VAL，保持本轮拟合参数，依protocol预设VAL合同验收，再冻结后报告已暴露TEST。

**代码与运行。** `crane_project/utils/port_reliability_scale_consistency_v1.py`、`crane_project/tools/run_port_reliability_scale_consistency_v1.py`、protocol/sources、`tools/run_port_reliability_scale_consistency_v1.sh`与21项新测试。新模块、连续状态与原消融相关共55项本地测试通过；Python3.8语法、脚本语法、输入3份SHA/完整配对/角色计数核对通过。此段写入时尚未开始真实拟合。服务器已有输入，无需新的图像推理或GPU。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
bash tools/run_port_reliability_scale_consistency_v1.sh 20261008_paired_train_v1
```

结果集中`work_dirs/port_reliability_scale_consistency_v1/20261008_paired_train_v1/`，包位于work_dirs根目录，包含本轮报告、参数、逐帧评分和日志，不含权重或ROI缓存。保留服务器既有暂存/工作区改动。结果、回传核验、提交与清理状态在实际执行后追加。

**实际执行与来源。** 源码`73a541b`已推送，服务器快进拉取后在mmrotljj执行，21项新测试通过；两臂CPU Newton分别6/5轮收敛，最大梯度约1.54e−13/2.36e−16，无新图像推理/GPU训练。辅助尺度缺失4帧，均在fit；标准全部2558输出，缺失辅助不删除主框/中心。服务器GitHub专属proxy指向失效127.0.0.1:17890，首次拉取失败；本次命令对该URL项设空并清空代理环境完成拉取，未改持久配置。服务器既有工作区/暂存SHA前后仍为`b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1`、`d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e`。此前未提交的连续状态代码及现有结果文档修改原样保留并一起同步，不声明这些已有修改全部来自本次。

**TRAIN probe事实。** 492帧全部输出，其中Real351、Sim141；尺寸正确477（Real336/Sim141）、错误15（全部Real，14帧在seq05、1帧在seq01）。工作点`risk_le=0.7893035475373201`仅为监督诊断，受seq05正确保留约束确定，不是新线上policy。

| 同485接受数，TRAIN probe | FA / FR / ED / CR | 最长FA | 最长FR |
|---|---:|---:|---:|
| 跨尺度候选 | 12 / 4 / 3 / 473 | 4 | 2 |
| 相同fit的3特征控制 | 13 / 5 / 2 / 472 | 3 | 1 |
| 冻结full_simple | 14 / 6 / 1 / 471 | 4 | 1 |
| score | 15 / 7 / 0 / 470 | 5 | 1 |

Real同344接受为候选12/4/3/332，对三控制FA13/14/15；seq05同104接受为11/4/3/93，对控制12/5/2/92、13/6/1/91、14/7/0/90。seq01候选FA1，其余视频FA0，Sim全部141正确保留，均未增加同数FA。seq05正确保留93/97=95.88%、Real332/336=98.81%、总体473/477=99.16%；全帧正确覆盖473/492=96.14%、接受覆盖485/492=98.58%。同CR473对照仍为FA13/14/15，接受486/487/488，不能把其接受数与485直接混排。同485理论最少FA8，候选仍多4个；没有消除高正确保留下的错误使用取舍。

**排序与范围。** probe总体错误AUROC候选0.89392，对重拟合/冻结/score为0.88274/0.88847/0.45129；Real为0.84960，对0.84544/0.84901/0.44841。seq05候选0.69809，低于同fit控制0.71134，高于冻结0.66642；不能称视频内排序普遍提高。Sim probe没有尺寸错误，AUROC不可评价，不记满分。fit自身工作点同1912接受为候选FA44，对重拟合44、冻结43、score53，并未稳定优于simple；fit的Real错误AUROC0.72732对重拟合0.69133，有信号但不替代留出验收。禁止将此TRAIN结果与之前完整VAL/TEST性能直接比较。

**失败、机制线索与决定。** 9项失败全部属于最长FR相对3个控制的all/Real/seq05重复统计层，来源是同一个事件：real_seq05帧511—512均为尺寸正确框，候选FR连续2帧，对控制最长1帧；另FR为452、534，ED为457、515、517。这次不同于旧删除相对尺寸实验的“连续正确拒错”，确实损失连续正确观测。两帧规范短/长边相对误差分别约+1.38%/−1.93%、+2.60%/−3.85%，跨尺度差异却约0.13177/0.16739；提示辅助视图不稳定会把正确主框判得更危险，但不是唯一根因。候选的连续FA4对同fit控制3也需披露。`TRAIN_CAPABILITY_FAILED_STOP`、selected_arm=null，不进入辅助VAL或TEST、不修改/部署policy，不扫描尺度/系数/门限或事后放宽连续FR约束。保留“有限排序信号但正确连续观测保护失败”的结论，不写成跨尺度完全无效或Real高FA已解决。未来需有新机制处理一致性与主框正确性的区别，才另立有限设计；本轮没有待补跑任务。

**三分量保护与独立复核。** TRAIN全部主框/score/输出、中心标志、方向风险/标志逐帧不变，中心输出2558/2558、命中2557/2558；probe492/492输出、中心命中492/492、全帧中心正确覆盖492/492。完整分域/逐视频三分量统计存report；方向资格/原policy保持，不因尺寸候选迁移或重拟合。本地核对源/input/receipt和3个产物SHA，复算2558行、144组方法状态及连续段、同CR计数和成对AUROC，直接重建两臂loss/gradient。风险浮点差≤9.99e−16；门限成员以精确服务器评分核对，无末位重选。`local_review.json`保存核验，结果文件路径如下：

- `…/train_check/report.json` SHA `40e6b7c2090425b2a70ad7d19959531f6437f894bcca0ab2ff929fa075df1775`；
- `…/train_check/models.json` SHA `4e757fcad73a630cf625ed22d4551d2f4c729046e2eed9f3021fa800fa2489d2`；
- `…/train_check/scored_TRAIN.jsonl.gz` SHA `f363f502f831a5396bb1782d8481d6e6e684fb67f616b14737f18eea68e6080f`。

回传包SHA `d17b8d0693af6e109d0f1457cd16bf63f64649b0a4e3a7d1e03164fdf1858306`，本地安全解包与服务器一致。分析已写入本节及结果文档第12节；仅本次两端包待按SHA核对删除，结果目录/日志/参数、旧缓存/旧包/权重保留。清理完成后追加最终状态。

**最终清理（已完成）。** 写入分析后，先把本地核验回执同步到服务器同一结果目录，SHA为`2ab41c6d5cadd1ecde2479eda340c9b56007f19ae45a32f55bea53f0b71124ed`；核对报告与本次包SHA后，已删除仅`work_dirs/port_reliability_scale_consistency_v1_20261008_paired_train_v1.tar.gz`的本地/服务器两份，并验证不存在。两端完整结果及旧证据保留，无待训练/推理/补TEST或待清理的本次包。原运行指令是已完成存档，勿同名补跑。

**文档归档约定更新（用户要求，2026-10-08）。** 本次跨尺度候选的最终结论为“有有限排序收益，但完整验收失败，不采用”，未完成VAL/TEST评价。已从可靠性主要结果文档撤回原第12节；本节完整保留合同、运行、指标、核验和失败原因。以后未通过验收、未采用的实验仅写入运行记录，不再追加到主要结果文档；不因此删除代码、逐帧结果或历史证据，也不追溯清理其他已有章节。

## 31. 主／辅助预测角色证据有限对照（2026-10-08，TRAIN验收失败，不采用）

### 31.1 当前授权、目标与来源

用户先授权有限设计，随后明确授权按该合同实现、检查、Git同步及服务器运行、回传分析并清理本次压缩包。本轮范围为`port_reliability_scale_asymmetry_v1`的缓存TRAIN能力检验；不重训检测器/几何头，不改原policy，不开展VAL/TEST评价。已有未提交文档修改保留；失败或未采用候选仅存运行记录，不进入主要结果文档。以下合同在候选拟合前固定。

主框是标准尺度最终M（B24＋midpoint σ1.5／epoch03）；辅助框是同一图像固定半尺度下同一M的最终预测，经原native sx/sy还原。当前正式simple没有辅助分支。辅助框不修复/替换/平均主框，也不是GT、B中间框或独立检测器。目标只有“用补充角色信息判断主框尺寸是否错误”，不是选择哪一框交付、修框或保证推断出两框的真实类别。

原跨尺度候选与正式policy保持第30节冻结身份，输入沿用原PINS和旧配对TRAIN；设计核对所读`scored_TRAIN.jsonl.gz` SHA为`f363f502f831a5396bb1782d8481d6e6e684fb67f616b14737f18eea68e6080f`，仍为2558个标准输出。下表为原结果复算，不是新模型收益：

| 规范双边均≤10%的组合 | fit | probe |
|---|---:|---:|
| 主对／辅对 | 1718 | 433 |
| 主对／辅错 | 166 | 44 |
| 主错／辅对 | 36 | 7 |
| 主错／辅错 | 18 | 8 |
| 主对／辅助漏检 | 4 | 0 |

fit共1942、probe492；probe15个主框错误中14帧在seq05，1帧在seq01。主框尺寸错误在fit/probe均仅出现在Real，Sim正确样本不提供错误辨识验收。两尺度参与过M训练；probe已暴露且本设计利用其失败分析，不是独立确认集，不构成新数据或新增错误类型。

### 31.2 唯一候选、输入及可微监督

**新旧信息的区别（待验证假设）。** 第30节5维候选已含主框三描述量，因此整体并非完全对称；问题是新增差异项只保留最大绝对量，丢失轴别/符号及辅助分数信息。本草案仅增加这些细节，不宣称获得独立质量证据。辅助score更高而尺寸更错的512帧排除了“直接选高score框”的依据，但不证明辅助分数在条件评分中完全无信息。

记规范原图长/短边为L、S，m/a分别为主/辅；score记s，按原simple规则夹至[1e−6,1−1e−6]后取logit。定义：

\[
\Delta_L=\log L_a-\log L_m,\quad
\Delta_S=\log S_a-\log S_m,\quad
\Delta_q=\operatorname{logit}(s_a)-\operatorname{logit}(s_m),
\]
\[
d=\max(|\Delta_L|,|\Delta_S|),\qquad
x=[\operatorname{descriptor}(m,\mathrm{image\_size}),d,h,\Delta_L,\Delta_S,\Delta_q].
\]

descriptor仍为score_logit、log_relative_geometric_size、log_aspect；h为辅助漏检。**保留原绝对差，增加3项，固定8维、9个线性参数（含bias）**，避免只用signed线性项丢失双侧变化幅度。辅助缺失时h=1，d/三个delta全为0，并保留主框所有分量；主框缺失则不产生风险，辅助存在也不生成新主输出。图像/两框的原图坐标须继承原管线，不再除以名义0.5；规范边排序处理w/h交换，角度π周期不进入尺寸输入，不改变方向接口。

这些信息仍是代理：正delta说明辅助更大，不证明主框偏小；负delta不证明主框偏大。给定原图尺寸与主三描述量，两个边delta可恢复辅助的尺度/比例描述，delta_q可恢复辅助score_logit；因此它等价于加入辅助框几何/分数细节，不是额外训练出的辅助质量。无新网络、独立参考、图像梯度、响应熵、时序、第三尺度、非线性交互或额外质量头。

**监督只针对主框。** TRAIN离线定义`y_m=1`当任一规范主边相对GT误差>0.1，否则0；辅助的正确性仅用于下文离线组合记账，不训练4分类，不选择可信框。拟合z=(x−mean_fit)/std_fit，std<1e−8沿旧规则置1，t=b+wᵀz：

\[
\mathcal L=\sum_i\alpha_{y_i}\,[\operatorname{softplus}(t_i)-y_i t_i]
 +\frac{0.1}{2}\|w\|_2^2,\qquad
\alpha_1=\frac{0.5}{N_{bad}},\quad\alpha_0=\frac{0.5}{N_{good}}.
\]

b不正则化。Torch实现若需要可直接使用softplus/BCEWithLogits，固定GT标签与缓存特征不求导，梯度仅作用b/w；本有限执行预定复用NumPy float64凸目标与解析gradient/Hessian的Newton求解，无需GPU/autograd或检测/几何梯度。在线风险sigmoid(t)是类平衡训练的排序分数，不是已校准错误概率；API仅接收主/辅预测、原图尺寸和冻结参数，绝无GT、domain、sequence、历史或真实组合类别。深度API不读辅助框。

### 31.3 固定对照、职责与工程预检合同

只允许拟合一次8维候选，与旧5维候选的完整可核验6轮末态作增量控制；旧5维未采用，作为实验控制不恢复其线上身份。另保留同fit的3维重拟合控制、冻结full_simple和score。旧5/3维来源与输入必须逐值/逐字节复核，不能只因维度相同假定是同一模型；必要时只做原固定求解器的数值重现，禁止新初始化、权重或门限扫描。候选从全0参数开始，原5维参数不能直接扩展作热启动；9参数对旧6参数的容量变化披露，不称等参数消融。

复用原fit1942（Real1377／Sim565）、probe492（351／141）、legacy64、purged60和2558对尺度，禁止重分组、定向重采样、添加扰动或依据错误替换帧。fit只排除主输出缺失，不以GT中心/正确性选帧；本源fit主输出1942，正确1888、错误54，辅助缺失4不删除。标准化、类别权重与全部模型参数只依据fit；L2=0.1、最多100轮、gradient∞norm≤1e−8、30次原固定Armijo回溯、bias不正则、float64。无其他固定系数、学习率、轮次选择或随机种子搜索；失败不扩大预算。

预检不得计算候选性能后再改输入：核对全部source/PINS、精确标准M回放、配对GT/角色、原图坐标恢复身份、主/辅助漏检、宽高交换/π周期、单位等比例不变性；8维固定schema及非法输入拒绝；控制5/3维风险重现；实际目标梯度有限差分、解析gradient/Hessian与一次更新loss下降；决定API逐帧只改size风险/flag，box/score/中心/方向/输出身份不变，GT永不进入在线路径。检查通过只证明工程合同，不证明收益。

### 31.4 预先继续／停止条件

1. probe使用一条全局、整tie接受的风险门限，以各域/视频正确保留≥95%确定，仅作监督开发诊断；不在GT组合内确定工作点，不部署此门限。fit/legacy/purged保留诊断，不选模型；不能用511—512两帧单独恢复作为门槛。
2. 在相同接受数下，probe总体与Real的FA必须严格少于**旧5维、同fit3维、冻结simple、score四种控制**，其他域/视频FA不得增加，控制ties用最少FA边界保守验收。各组最长FR不得超过任一同数控制，沿用第29/30节评价职责，不因新设计放宽；同时报告同CR下FA/覆盖及各FA/FR/ED/MISSING/总不可用段。
3. 新增组合保护仅比较候选与旧5维以验证此增量：先在每个完整domain/video的全部主输出中按GT-free风险及原图像tie次序选定同接受数集合，再用GT做组合记账。“主对／辅错”FR不增加；“主错／辅对”及“主错／辅错”FA各不得增加。四种组合和辅助缺失的数量、FR/FA、比例全报，无支持项记不可评价；不能在GT子组里重新排序、按GT路由、只展示有利子组。总FA严格改善与三组合保护同时满足，不能用救回正确框交换更多共同错误。其余3控制也报告组合计数，但不新增事后逐组阈值。
4. 所有分组均保留score/simple同数控制，中心只统计输出帧命中并另报输出覆盖及全帧中心正确覆盖，方向保持原资格/决定并单列分母；只变化size判断。工程/任何继续条件失败立即结束本版本，不改特征、阈值、正则、预算、采样或模型，不启动辅助VAL推理/TEST；失败仅写运行记录。
5. 全部通过最多证明当前已暴露TRAIN存在有限增量，不能宣称Real高FA解决或泛化。须先保存参数/合同/source SHA再提出完整VAL阶段；本次设计未授权该阶段运行，也没有TEST任务。已暴露TEST只能在候选/门限依据TRAIN/VAL冻结后报告，绝不用于挑选本设计。

**风险与本轮完成状态。** 当前只有15个probe主框错误且集中一个视频，已知失败参与设计，过拟合风险显著；同模型两个已训练尺度共同出错不能靠关系量保证检出，辅助score不能担保尺寸，Top1可能不是同一物体。即便未来CPU缓存验证通过，线上仍需额外一次B＋M推理并单独量时。以上为项目适配假设，不是已验证的新方法；没有足够证据保证降低真实VAL/TEST FA。已完成8维实现、工程预检、服务器固定一次拟合及本地独立复核，结果为TRAIN验收失败；本候选和旧跨尺度版本封存，未迁移policy。原5/3维风险以≤1e−12容差重现，历史门限比较始终读取精确旧评分，避免跨平台libm/BLAS末位影响tie成员。

### 31.5 实现、检查与运行路径

新增（未修改旧算法/旧源清单）：`crane_project/utils/port_reliability_scale_asymmetry_v1.py`、`crane_project/tools/run_port_reliability_scale_asymmetry_v1.py`、同名protocol/sources JSON、`tests/test_port_reliability_scale_asymmetry_v1.py`、`tools/run_port_reliability_scale_asymmetry_v1.sh`。命令及合同均在Git内，结果集中同一版本父目录，压缩包在work_dirs根目录。入口没有检测推理或VAL/TEST分支；仅拟合一次候选，旧模型4份产物（含completion）先核对字节SHA/源/输入/逐帧身份，再复用冻结评分。辅助缺失不删主框，主缺失不接受辅框；仅size风险/标志改变。

**本地预检（事实）。** 25项新增测试通过；Python3.8与shell语法通过。2558对主辅框、GT/角色与精确标准M身份一致；1942个fit有54个主框错误、1888个正确。旧5维特征最大末位差8.89e−16，5/3维风险最大差1.00e−15/2.78e−16；正式simple与score差0，使用精确旧评分比较。实际8维目标gradient/Hessian有限差分最大差3.54e−11/8.00e−10；零初始化单次Newton更新loss从0.693147降至0.575557，仅为梯度工程预检，不是候选性能或完整拟合。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
bash tools/run_port_reliability_scale_asymmetry_v1.sh 20261008_role_evidence_v1
```

产物：`work_dirs/port_reliability_scale_asymmetry_v1/20261008_role_evidence_v1/train_check/`；包为`work_dirs/port_reliability_scale_asymmetry_v1_20261008_role_evidence_v1.tar.gz`。脚本拒绝覆盖，CPU运行且禁用GPU；保存测试/工作区及暂存diff SHA、报告/参数/2558逐帧评分/完成回执。先回传核验、写结果，再仅删除本次两端包；所有结果/旧缓存/旧权重保留。本指令已执行成功，是存档复现指令，勿同名补跑；验收失败不构成待追加训练任务。

### 31.6 实际执行、结果、核验与决定

**执行事实。** 源码提交`6a9baed46422bb9b644db9917f2a69cfbf07ca86`在本地/服务器均使用`main`，已推送原origin并快进拉取；不是其他分支或新工作树。初次push自动审批因具体远端未核验而拒绝，核验两端既有origin均为`https://github.com/lusirs123/mmrotate-crane1.git`、上次共同提交均为`8d82b0e`及上传范围后，原动作获准并成功。服务器拉取复用临时loopback代理转发，未改持久配置。25项新增服务器测试通过；唯一候选float64 Newton共6轮收敛，objective从0.693147降至0.5698368709512137，最大gradient约8.03e−12；旧5/3维模型没有重拟合。服务器原工作区/index改动SHA前后仍为`b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1`／`d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e`。无新图像推理、GPU、检测/几何训练或深度计算。

**TRAIN probe事实。** 492帧全部输出，477正确、15错误（全Real、14帧seq05）；本轮8维全局监督诊断点`risk_le=0.7919525125386913`，各域/视频正确保留均≥95%，不是线上policy或独立标定。候选接受483（98.17%），正确保留472/477=98.95%、接受风险11/483=2.28%、全帧正确覆盖472/492=95.93%。以下控制都在各自完整统计组中先按原评分及GT-free图像tie次序选择同数集合，之后计数；不能与第30节同485接受数表直接混排。

| probe同483接受数 | FA / FR / ED / CR | 最长FA | 最长FR |
|---|---:|---:|---:|
| 新8维角色证据 | 11 / 5 / 4 / 472 | 3 | 2 |
| 冻结旧5维跨尺度（未采用控制） | 12 / 6 / 3 / 471 | 4 | 2 |
| 相同fit的3维控制 | 13 / 7 / 2 / 470 | 3 | 1 |
| 冻结full_simple | 13 / 7 / 2 / 470 | 4 | 1 |
| score | 15 / 9 / 0 / 468 | 5 | 1 |

Real同342接受为新候选11/5/4/331，seq05同103为10/4/4/93；其余视频FA不增加，Sim141/141正确保留但无错误样本，AUROC不可评价。同CR472时控制FA12/13/14/15，接受484/485/486/487，均为独立同CR比较。新候选probe pooled/Real/seq05 AUROC为0.91125/0.87401/0.74742，对旧5维0.89392/0.84960/0.69809，有有限排序信号；15个错误集中一个视频且probe已暴露，不能宣称泛化。

**组合保护事实（probe完整组同483接受数，非GT子组工作点）。**

| GT组合 | 支持数 | 新8维 | 旧5维同数控制 |
|---|---:|---:|---:|
| 主对／辅对 | 433 | FR 0，CR 433 | FR 2，CR 431 |
| 主对／辅错 | 44 | FR 5，CR 39（FR 11.36%） | FR 4，CR 40（FR 9.09%） |
| 主错／辅对 | 7 | FA 4，ED 3 | FA 5，ED 2 |
| 主错／辅错 | 8 | FA 7，ED 1 | FA 7，ED 1 |
| 辅助缺失 | 0 | 不可评价 | 不可评价 |

减少了“主对辅对”误拒及一部分“主错辅对”错误使用，但“主对辅错”FR净增1，共同错误仍接受7/8。不能以总体FA/FR都更少替代组合保护；也不能由此证明辅助score或符号完全无用。fit同1910接受为42/20/12/1868，对旧5/3维FA44、simple43、score53；fit有少量改善，不改变probe失败决定。fit四个辅助缺失主正确框全部保留。

**失败与机制解释。** 11项失败包括9项最长FR（all/Real/seq05相对3个控制，重复指向同一段511—512）及2项“主对辅错”FR保护（all/Real重复统计层），不是11个独立错误事件。新候选FR为seq01帧296、seq05帧452/488/511/512，五帧全部主对辅错；ED为seq05帧453/454/457/515。511—512主框尺寸最大误差仅1.93%/3.85%，辅助最大误差11.88%/13.67%，仍被连续拒绝。**已确认：增加符号、轴别及辅助分数提高了部分排序，却没有建立合同要求的主辅错误区分。推断：关系代理仍把辅助不稳定传到主框风险。待验证：这不是线性表示、特征信息或支持分布中某一项已查明的唯一根因。** 不扫描门限/正则、不给511—512定向补丁、不追加预算或放宽合同。

**保护与本地独立核验。** 2558行主/辅预测、原图尺寸、GT、角色、score及源/input/旧控制/回执产物字节SHA通过。所有原主框、中心/方向风险及标志保持；中心仅输出帧统计，TRAIN命中2557/2558、输出覆盖2558/2558、全帧正确覆盖2557/2558，probe三项均492/492；方向资格和分母沿原policy并另报。独立公式核对8维特征/风险最大差9.16e−16/1.84e−15；精确服务器评分复算324份状态/连续段（含同CR）、180组AUROC/AP、整tie门限、全组同数tie边界和全部组合，11项失败逐一重现；最终objective/gradient独立重建通过，无重新拟合。`local_review.json` SHA `dbf10c358e5ca0ee60d27c1b31f9a06cc7667baf69fc54689dc649f0d8a9e58c`，回传包SHA `31d1569f8515717f3effd0e76627c16e1e280f6d6269d7ea204cd9422bf2e2f5`，安全解包路径/类型检查通过。

核心产物均位于本节31.5结果目录：`report.json` SHA `f4fd0dd94f5279d84c7f8e1b4d1e7817f372ada5cda34df0a4beb66fdb0a3908`；`models.json` SHA `3126d2c45504bc705a88a88c0a0f3727ba721e5c2e54f57d1179fb108864ee72`；`scored_TRAIN.jsonl.gz` SHA `33e53d8c1cd52938b8b11aada2e0214aaed0382da4fc804baabfa6ab6dfaf2b7`。完整日志、合同、工程检查、各域/视频/角色及组合支持存报告，不扩展主要结果文档。

**决定与待办。** `TRAIN_CAPABILITY_FAILED_STOP`、selected_arm=null，保持正式M/simple/policy；本版本不采用、不进入VAL/TEST。收束此次跨尺度线性评分增量，不继续自动搜索特征、权重或预算；下一项须先说明新的主框尺寸正确性证据及有限合同，当前没有待训练任务。该结论否定本候选，不能推广为所有主辅策略无效。**清理已完成：** 分析写入后，回执已同步服务器且SHA一致，按报告/回执/包SHA核验后，仅删除本次两端`work_dirs/port_reliability_scale_asymmetry_v1_20261008_role_evidence_v1.tar.gz`，两端均验证不存在。完整结果、模型参数、日志、缓存/权重及旧证据保留，无本次待清理包。源码与最终记录均在main同步。

## 32. 标准尺度主框对边局部尺寸证据：有限合同与执行（2026-10-08）

**任务与身份。** 用户授权本地设计/实现/验证，仍在main提交推送，服务器同项目快进拉取并执行，回传核验和分析写入后仅清理本次两端压缩包。收束第30/31节跨尺度5/8维；本项直接判断固定B24＋σ1.5/head_epoch03交付主框M的两边是否均≤10%，只改size风险/flag。中心、方向、score、原框/输出/漏检身份、正式simple/policy保持；不修框、不重启深度、旧ROI头、尺寸训练或TEST选择。以下采样、求解、门限及继续条件在本项拟合/性能计算之前固定。

### 32.1 缓存可达范围与固定采样

服务器核实`work_dirs/port_geometry_midpoint_formal_v1_roi_cache`仅保存B对齐的256×9×9旋转ROI及支持掩码，没有完整P3；全部2558帧缓存B与原标准尺度历史B逐值一致（各坐标/边/角/score最大差0）。复用原train_s1及条件val_s1，禁止读s05/新尺度。旧普通ROI确为3×3空间池化，不能声称它只有全局平均。

同一缓存每个位置固定提取跨256通道的mean与RMS两份标量场，两臂共用，不学习通道投影或特征网络。原图M的规范长/短轴查询通过**实际sx/sy、缓存B的模型规范轴和原raw宽高—角关系**映射到存储网格；绝不再除以名义倍率。缓存9×9点覆盖原B的1.5倍范围，模型侧最小16px，与原capture一致。越出缓存或图像支持时显式记录0，不外推或称插值提高分辨率。

- 普通ROI控制：M的1.5倍规范裁剪上固定9×9查询，分成真实3×3区域，每区mean/RMS/支持率共27项，加simple原3项为30维、31参数。
- 对边候选：长/短法线×正/负侧×内/边/外三带，每带沿另一轴全长的−0.35/0/+0.35位置查询，聚合mean/RMS/支持率，共36项，加原3项为39维、40参数。轴、侧和带位置保留，不先重建参考尺寸。
- 三带法向间距由源分辨率唯一确定：至少一个缓存网格间隔（L∞）且至少一个模型P3步长8px，不依据GT误差选择。若间距超过M半边，内带将越过中心，该轴证据整体置0并披露未分辨；不删帧、不用GT中心/正确性选帧。至少一个fit帧须有两轴分辨且12带支持均>0，否则工程上停止、不开拟合。

这是**同一压缩信号空间组织的有限检验**，不是历史256通道ROI重现或等参数消融；30/39维容量不同须披露。mean/RMS可能丢失语义，9×9不足以保证辨识10%误差，可见实体边不必对应固定比例标注边。不同波段有采样间距不等于有真实物体边界，不保证候选有效。

### 32.2 唯一拟合预算、数据职责与停止条件

沿用fit1942（Real1377/Sim565，错误54/正确1888）、probe492（351/141，错误15全Real且集中seq05）、legacy64、purged60；只取标准M主框字段，丢弃辅助预测/特征，所有主输出均参与相应角色。原同fit3维控制读取完整旧模型与精确旧风险，数值重现容差1e−12，不再次拟合。正式full_simple/score另报，普通ROI只作控制，不作为备选方案。两新臂各唯一一次float64零初始化、fit均值/标准差、类别各0.5、L2=0.1不惩罚bias、Newton最多100轮、gradient∞norm≤1e−8、30次原Armijo回溯；无网络、随机搜索或参数续训。

**与旧第31节不同：本项所有5种方法的全局整tie比较门限只在fit上按各域/视频正确保留≥95%确定一次。** probe/legacy/purged及条件VAL只能套用，不重标定，原线上policy不动。这避免在已暴露probe上重新选择工作点，不能拿旧483接受数表与本项不同固定点直接比较。

继续合同：probe每域/视频正确保留≥95%；同接受数总体与Real FA严格少于同fit3维、普通ROI、冻结simple、score四控制，其余域/视频FA不增加，ties以控制最少FA边界保守核验；同CR下FA沿同样要求；每组最长正确误拒FR不得超过任一同数控制。FA/FR/ED/CR/MISSING分开计数，连续FA/FR/ED/MISSING/总不可用均报告，总不可用不能因正确拒错而单独否决。相同接受数FA改善等价于CR/全帧正确覆盖保护，不能以多拒绝换取表面低风险。

工程/任何probe门槛失败即停止，不换采样、维数、损失、正则、预算、门限；不跑VAL/TEST。**仅全部probe通过才允许同一冻结模型和fit门限评价完整VAL887帧**，该阶段不拟合或重新定工作点，仍按合同保护各域/视频；即使通过也不自动部署或进入TEST。本项用户授权这一条件执行，尚无独立泛化保证。probe错误少且已暴露，不能宣布通用/安全保证，类平衡sigmoid不是已校准概率。

### 32.3 文件、检查与运行路径

新增`crane_project/utils/port_reliability_opposite_border_v1.py`、对应tools入口、protocol/sources JSON、`tests/test_port_reliability_opposite_border_v1.py`、`tools/run_port_reliability_opposite_border_v1.sh`。旧源清单、算法、数据/policy、权重/论文不改。源/缓存/控制四产物/原policy全部SHA锁定，只有字节固定的仓库缓存可由Torch载入。

本地30项必要测试通过（增加GT配对精度测试）：真实3×3分区、内边外网格间距、未分辨短边不删帧、通道统计、越界/掩码、实际sx/sy与宽高关联、π周期/规范交换及等比例原图单位不变、固定schema、目标gradient/Hessian有限差分/更新、漏检不判断、只改size、fit限定门限与整ties/逐视频要求、失败禁止VAL、路径/覆盖保护及源清单校验。实际数据控制最优解/风险、实际两新臂导数与零初单步下降须服务器拟合前再次验证。测试通过仅证明工程路径。

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
bash tools/run_port_reliability_opposite_border_v1.sh 20261008_border_evidence_fix1
```

CPU缓存分析禁用GPU，无需后两张卡。版本结果共置`work_dirs/port_reliability_opposite_border_v1/20261008_border_evidence_fix1/`，下分train_check及仅在通过时的val_check；包在work_dirs根`port_reliability_opposite_border_v1_20261008_border_evidence_fix1.tar.gz`。脚本拒绝覆盖，保存原服务器工作区/index SHA、测试/运行日志、source/合同、模型/固定点、全部逐帧风险/决定及固定场NPZ，便于本地独立复算采样；结果不是需Git同步的源码。本指令已执行成功，为存档命令，勿同名补跑。分析写入32.4后仅清理本包，两端原缓存/所有结果/权重保留。

### 32.4 实际执行、回传核验与决定

**工程与执行事实。** 首轮源码3c7338c、运行20261008_border_evidence_v1的29项服务器测试通过，但拟合前GT角逐值检查停止。全部缓存中心与双边完全一致，2167帧仅角度因float32/le90周期还原不同，最大差1.78814e−7rad。修复限定中心/尺寸仍逐值相同，仅允许周期角差≤2e−6rad；保持原GT/分量资格/决定，角度不进入尺寸特征。增加配对测试，不修改采样/预算/性能门槛；保留首轮日志。唯一实际拟合源码`e35e560eebe935441b33a458cef013356c7f4e3f`，两端main，目录`20261008_border_evidence_fix1`。TLS拉取中断重试一次后快进成功，无持久配置改变；30项服务器测试通过，CPU运行，无新图像推理/GPU/检测或几何更新。服务器原worktree/index SHA前后仍为`b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1`／`d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e`。

缓存别名实际解析到`work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache`，原SHA不变。TRAIN2558帧保留，两轴分辨2556帧、12带均有支持1463帧；fit为1940/1942及978/1942（Real完整支持716/1377，Sim262/565），probe为492/492及411/492（Real270/351，Sim141/141）。2个未分辨短轴fit帧显式0输入仍保留，不依据GT筛选。工程覆盖条件通过，不代表足以区分10%误差。

普通ROI／对边两臂实际gradient/Hessian有限差分最大差1.69e−11/3.88e−10、2.63e−11/5.59e−10；零初单步objective由0.693147降至0.553621/0.541461。各唯一拟合5轮收敛，最终objective0.5506283750953165/0.5379194482355177，gradient∞norm2.47e−10/1.75e−9。原3维控制未重拟合。这些是工程/拟合结果，不能替代判断效果。

**冻结fit工作点与probe事实。** 对边门限`risk_le=0.8309459057160208`只在fit确定，不使用probe定点。probe492帧全部输出，477正确/15错误（全Real、14seq05）。候选接受475，FA12/FR14/ED3/CR463，接受风险2.53%、正确保留97.06%、全帧正确覆盖94.11%。Real正确保留322/336=95.83%，但seq05仅83/97=85.57%，故总体≥95%不能替代逐视频约束；14个FR全部在seq05，均两轴分辨且12带有支持，不是缺失证据帧。最长FR3帧，两段532—534、536—538；其余FR段详见报告，正确拒错不纳入FR。

| probe同475接受数 | FA / FR / ED / CR | 最长FA | 最长FR |
|---|---:|---:|---:|
| 对边39维候选 | 12 / 14 / 3 / 463 | 3 | 3 |
| 同fit3维simple控制 | 11 / 13 / 4 / 464 | 3 | 5 |
| 普通3×3 ROI 30维控制 | 11 / 13 / 4 / 464 | 2 | 5 |
| 冻结full_simple评分 | 12 / 14 / 3 / 463 | 3 | 5 |
| score | 15 / 17 / 0 / 460 | 5 | 2 |

同CR463时四控制分别FA11/11/12/15、接受474/474/475/478，对边没有比simple/普通ROI减少错误使用。总体probe AUROC对边0.86150、同fit3维0.88274、普通ROI0.88232、冻结simple0.88847；Real为0.80456/0.84544/0.83413/0.84901；seq05为0.49632/0.71134/0.54050/0.66642。score总体/Real/seq05为0.45129/0.44841/0.38660。Sim无错误，AUROC不可评价，不能宣布Sim辨识改善。

fit总体AUROC对边0.82869高于3维0.78319，但同1919接受数对边FA51，3维/普通ROI/simple为44/49/44；拟合loss或整体AUROC改善没有转成所需工作点收益。原正式simple/policy保持不变；本表是同数控制，各方法新fit点也只是比较。套用各自fit点时seq05正确保留为候选83/97、3维88/97、普通ROI81/97、冻结simple86/97、score94/97，说明fit工作点转移本身也是限制，不能把这当作正式policy既有结果。**不在probe上重新定门限，且候选排序弱于简单控制，失败不能只归于门限。**

**失败、边界与决定。** 19项记录为8项同数FA、8项同CR FA、2项最长FR（相对score，all/Real重复）、1项seq05正确保留；统计层重复不代表19个独立事件。对边同数FA没有少于simple，seq05误拒显著，不能采用普通ROI或放宽门槛补救。已确认此固定mean/RMS＋9×9对边空间读出没有稳定增量；推断是局部信号在当前表示/线性学习中未对应主框尺寸正确性，尚不能确定原特征、统计压缩、线性容量或错误支持中的唯一根因。不能推广为所有单尺度边界证据无效，也不能声称改善深度或机械安全。

结论`TRAIN_CAPABILITY_FAILED_STOP`、selected_arm=null；原合同阶段未运行VAL/TEST，不继续特征/权重/门限/预算搜索。后续用户另授权第33节冻结VAL补评，不追溯更改此结论。正式M、simple和policy保持，旧跨尺度与本项均封存；普通ROI只作控制，不采用。下一步先收束现有可靠性结果用于论文的任务/协议与固定基线表达，不自动开新头；若另立候选，须先有区别于当前压缩缓存的直接主框正确性证据和有限合同，不仅重复空间重排。失败只写本运行记录，主要结果文档/论文不追加本候选成果。

**回传独立复核。** 包SHA`e52fe79c2e79f699deb509757e3f22ded736a3a0b1cf4c6bb48faf34635116be`、tar路径/类型安全检查通过。2558行原M/B/GT/角色、score/输出身份及只改size决定逐帧一致，原方向标志/风险/资格保持；probe中心输出帧命中492/492、输出覆盖492/492、全帧中心正确覆盖492/492，方向另沿原449可评价输出分母。全部原缓存、source/input/控制/回执SHA核验通过。用独立标量双线性、区域/边带循环与公式重建特征最大差8.88e−16、风险9.55e−15、最优objective2.22e−16；344份四状态/连续段（含同CR）、180组AUROC/AP、所有fit-only整tie工作点、同数tie边界及19项失败逐一重现，未重新拟合。完整256通道缓存未复制本机，channel mean/RMS由字节固定源/缓存和已知值测试核验；本机固定场NPZ可独立重建空间读出，不能声称复制了完整P3。

核心产物在本节结果目录`train_check/`：`report.json` SHA`bdf70c2ca312a02d3abf93bf9790b1075e4f6e52a0fe9e72ce592de5c9228795`，`models.json` SHA`b533fd0efb2849ac22d449c3302aa398f419813d8d2c04330c003e9156c084ef`，`scored_rows.jsonl.gz` SHA`3bf194e0bb17cf7d8e4513e6ac3259a494d8dbb9e3561473fa02e025d2f7fef0`，`fixed_fields.npz` SHA`59a9e88dec568e9d469e50dc0bb08f682fdb92bc1ae6ea3d6d1275c2a08fc73d`；同父`local_review.json` SHA`a90aedd99ba39cf4032b66ac21f040af9820c58a45a3595949598193a6eb1b2e`。源码/最终记录沿main同步。清理已完成：分析写入后回执同步服务器且SHA一致，经报告/全部产物/包SHA核验，仅删除本次两端fix1包，两端均验证不存在；结果、参数、固定场、原缓存/权重、首轮工程日志全部保留，无本次待清理包。


## 33. 冻结VAL补评与后续统一TRAIN／VAL／TEST协议（2026-10-08）

### 33.1 本次授权、冻结边界与实现

用户明确授权：第32节现有冻结模型即使probe失败，也补完整VAL评价，另记“冻结VAL补评”；不重新拟合/调整采样或fit门限，不追溯原合同通过，不根据VAL补评扩大预算/选新结构。后续新候选只使用现有TRAIN、VAL、TEST，不再增加数据角色，详见文首最新协议。当前模型仍来源原1942帧历史拟合集，不能称其已经按完整TRAIN2558重拟合；本次普通ROI和同历史训练子集3维控制复用原权重，正式simple和score另报。

新增单用途`crane_project/tools/run_port_reliability_opposite_border_v1_frozen_val.py`、source SHA清单、`tests/test_port_reliability_opposite_border_v1_frozen_val.py`及`tools/run_port_reliability_opposite_border_v1_frozen_val.sh`。不修改第32节封存源清单、训练/采样函数和失败回执；旧入口的失败禁VAL保护仍成立，新入口明确只接受用户授权的特定失败模型/报告/完整产物SHA。该入口无TRAIN拟合或TEST阶段、无拟合/定门限调用。旧普通ROI/对边/3维模型、5种fit门限逐值读取，原M/GT/score/中心/方向决定保留，唯一变化size风险/flag。正式policy的原尺寸标志另报，避免把新fit比较工作点当成正式policy。

VAL仍为原887帧（Real375、Sim512），886帧有输出、1帧无框保留；无框不产生可靠性风险或FA/FR，覆盖及MISSING单列。使用原字节固定val_s1缓存及原VAL主框，actual sx/sy、规范轴和原raw宽高关联沿第32节。返回当前M对边三带、普通ROI3×3以及simple/score，完整分域/逐视频报告FA/FR/ED/CR、接受风险/正确保留/全帧覆盖、错误接受/正确误拒/正确拒错/漏检/总不可用连续段，同数和同CR比较含保守tie边界。原95%保留与控制保护仅作冻结点描述检查，不叫旧合同续跑或自动采用；不扫描VAL门限。补评后再根据完整证据做保留/结束决定。

本地9项测试通过：限定特定封存模型/失败身份、改字节/任意模型拒绝、旧保护不变、VAL唯一数据角色、运行时禁止拟合/门限求解、工作点不重算、仅size改变/漏检无风险、正式policy单列、路径/覆盖保护、887帧来源与源闭包。Python3.8和shell语法检查通过；无新特征、网络、训练预算或GPU任务。

### 33.2 服务器命令与产物

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
bash tools/run_port_reliability_opposite_border_v1_frozen_val.sh
```

结果置同一实验父目录`work_dirs/port_reliability_opposite_border_v1/20261008_border_evidence_fix1/val_supplement/`，原train_check和VAL_SKIPPED.txt保持历史阶段身份。包在work_dirs根`port_reliability_opposite_border_v1_frozen_val_20261008.tar.gz`，拒绝覆盖。保存源/模型/固定点SHA、report、逐帧scored_VAL、固定场NPZ、模型原字节副本、日志/测试和原服务器diff SHA；本指令已执行成功，为存档命令，勿同名补跑。回传核验/写分析后仅删除本次两端包，全部产物、旧权重/缓存与失败日志保留。源码/记录在main上传同步，不改原dirty工作区/index。

### 33.3 实测、核验与决定

**执行事实。** 源码`69329adf0b050badf9cd07dff2c33e9c5b70b62c`在两端main同步，服务器CPU成功完成全部887帧VAL，9项新增测试通过。现有模型原字节SHA与第32节相同，5个fit门限逐值相同，fit/control/cutoff更新均0，没有新推理、特征/网络训练或TEST访问。原probe失败回执/19项记录不变，补评身份`FROZEN_VAL_SUPPLEMENT_COMPLETE_REVIEW_REQUIRED`，原合同仍未通过，不迁移policy。

VAL输出886/887，原正确尺寸663帧（Real199、Sim464）、错误223帧（175/48）；缺失1帧不进入四状态。候选冻结门限0.8309459057160208，接受884，FA221、FR0、ED2、CR663；接受风险25.00%、正确保留100%、全帧正确覆盖663/887=74.75%。仅seq07帧19/180两个尺寸错误被拒，Sim48个错误全部接受；最长FA13帧、FR0。每域/视频正确保留均100%，但高保留不意味着识别错误有效。

**公平比较。** 同884接受数下所有4控制FA也为221、FR0、ED2、CR663；此点FA理论下限884−663=221，不能凭持平宣布排序等效，也不能拿不同接受数的原fit点比较直接判优。保留同样全部663个正确框、采用各控制最小整tie接受集合时：

| 完整VAL同CR663 | 接受数 | FA | ED | FR |
|---|---:|---:|---:|---:|
| 对边39维候选（冻结fit点） | 884 | 221 | 2 | 0 |
| 同历史训练子集3维控制 | 874 | 211 | 12 | 0 |
| 普通ROI30维控制 | 878 | 215 | 8 | 0 |
| 冻结full_simple评分 | 871 | 208 | 15 | 0 |
| score | 872 | 209 | 14 | 0 |

这些控制同CR集合是离线GT评价下的公平排序参照，不修改线上/历史fit门限。候选比simple多13个FA且正确保留完全相同，未改善错误使用。Real同CR199为候选FA173、3维166、普通ROI167、simple161、score163；Sim同CR464为48/40/44/45/44。seq07同CR63为160/144/154/141/150；seq14同CR136各方法FA均13，没有稳定尺寸辨识。

| 错误AUROC | 对边 | 3维控制 | 普通ROI | 冻结simple | score |
|---|---:|---:|---:|---:|---:|
| pooled | 0.74437 | 0.78216 | 0.78908 | 0.78563 | 0.68439 |
| Real | 0.56864 | 0.55922 | 0.64256 | 0.54478 | 0.66475 |
| Sim | 0.53390 | 0.67767 | 0.64776 | 0.69338 | 0.69513 |
| seq07 | 0.47090 | 0.51675 | 0.47835 | 0.53919 | 0.46825 |
| seq14 | 0.31618 | 0.31052 | 0.57749 | 0.41516 | 0.48416 |

Real总体AUROC相对simple略增，不能抵消同CR FA退化、seq07及Sim排序弱化；不把失败归给唯一视频。普通ROI控制也有部分排序信号，但它未作为本项候选，不事后转而采用或重新定门限。28项原保护条件的**描述检查**为8项同数FA严格改善未达与20项同CR FA退化（all/domain/sequence重复统计层），无FR/正确保留失败；不是新合同通过/续训决定，也不是28个独立错误事件。

**正式policy另报，身份不变。** 原正式simple尺寸标志为FA190、FR10、ED33、CR653（接受843，Real144/10/31/189、Sim46/0/2/464），这是原policy工作点，不是表中同CR参照或新fit门限。候选与正式policy接受数/CR不同，不能直接用221对190宣称等工作点优劣；上表同CR663和完整排序已足够说明未形成稳定增量。中心只在输出帧计算命中874/886=98.65%，另报输出覆盖886/887=99.89%、全帧中心正确覆盖874/887=98.53%；Real362/374、Sim512/512。方向沿原887可评价帧、886可评价输出分母，风险/标志/资格逐帧不变。可靠性拒尺寸不删中心，也不构成深度/安全保证。

**独立核验及决定。** 两轴分辨886/886输出，12带均支持422/886，未以GT过滤；VAL缓存/原主框/GT/score/policy源SHA一致。包SHA`a51caa3b7d50b784ab46369d190a1fb779e2b48fca20ab09f9a5a8a21713bccb`与安全tar路径/类型检查通过。887行独立标量双线性/3×3区域/对边循环重建特征最大差4.44e−16、模型风险9.88e−15；132份状态/全部连续段、60组AUROC/AP、同数tie边界、同CR集合、5个冻结工作点、正式policy及28项描述失败逐一重现。原失败报告和参数字节保持，fit/cutoff重算0。服务器原worktree/index SHA前后为`b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1`／`d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e`，无改动覆盖。

结论：完整VAL也未优于simple，问题不限原seq05；**结束本项固定9×9 mean/RMS对边线性方案**，不增加特征、训练网络或预算、不调VAL门限、不进入TEST。正式M/simple/policy保持；失败只记本运行记录，不写成主要结果/论文新成果。它不能否定所有主框局部证据，尚无唯一根因证明。下一步整理正式可靠性任务、评价协议及已采用结果进入论文写作；如将来有明确授权的新候选，统一完整TRAIN拟合＋完整VAL评价/门限＋冻结TEST报告，不再新增数据子角色，本次历史权重不重拟合或改身份。

产物均在33.2补评目录：report SHA`861e6b60f5a04cbe1b16edffb04e5f7f8c351d39f411f0cf06681204c416f7e8`，scored_VAL SHA`c96c6279bd35cd99f8978a91b39cc9500ac9254fbd1b8ed2438ef1356194aeaa`，fixed_fields SHA`621e69fd5efa29b47b37984ad238750c592d3aeacc13630de5d82c38449defd1`，frozen_fit_points SHA`483238611bd2a9c1dd66ad2ecf6bef1ad4a7903743d1c4a6bac296e68438371a`，frozen_models SHA仍`b533fd0efb2849ac22d449c3302aa398f419813d8d2c04330c003e9156c084ef`，local_review SHA`c08e235eaa9a9474c294249a570737814eb71b1e876797ab811a2266ed7c6de2`。清理已完成：分析写入后复核回执同步服务器且SHA一致，经全部结果/原参数/包SHA核验，仅删除本次两端冻结VAL包，两端均验证不存在。完整结果/模型/固定场与旧缓存/权重保留，无本次待回传或清理任务；源码/最终记录沿main同步。


## 34. ReDC启发的冻结几何特征尺寸评分（2026-10-08）

### 34.1 授权与预先固定设计

用户授权按已审议设计完成本地代码、必要检查、Git提交/服务器拉取运行、回传分析和压缩包清理。只研究可靠性尺寸评分，不修框，不恢复尺寸/深度支路。未核实作者官方GitHub，依据ReDC正文公式9和附录CR机制做项目适配；不是原论文CAR/四水平坐标校准复现，也不将类平衡风险称为真实错误概率。

固定当前B24＋σ1.5/epoch03与cachefix1原policy（SHA38d914…）；完整TRAIN2558（Real1810、Sim748）全部拟合，完整VAL887评价/定点，历史fit/probe/legacy/purged字段不进入新训练筛选。只读标准尺度缓存，记录原B→M、GT/身份配对，保留原M输出；跨设备重放只作来源核对，像素atol5e-4/rtol2e-5、π周期角2e-6rad、score逐值相同，并要求尺寸正确标签一致。中心、方向标志、原框/score/count/missing均保留。

实际冻结midpoint stem32×9×9→逐通道固定3×3均值池化288维，加最终M相对尺寸/比例2维，共290维温度输入；score_logit单独使用。ReDC臂290→16→8→1、ReLU，T=.25+softplus(a)，risk=sigmoid(-(z/T+β))，4802参数；线性控制完整291维、292参数。两臂neutral score-risk初始化，末层零权重；标准化只用完整TRAIN。唯一监督为原图规范长短边最大相对误差>10%二值标签；类别各总贡献.5，minibatch权重固定N/(2*Nclass)，不按当前批次类别重归一。没有其他辅助损失/增强/重采样。

固定seed1701、各100轮、batch256、各1000更新，同批次次序；AdamW lr.001/weight_decay1e-4仅weights，clip5。独立两步smoke丢弃，原头无梯度，最后第100轮权重唯一评价，不选最好一步。工程通过后完整VAL，不加probe性能预门槛。VAL为开发/门限数据，不包装成独立无偏测试。

每个评分仅一个全局VAL最小整tie门限，使每域/视频正确保留>=95%。原policy固定点另报；同接受数用图像ID分同分诊断并报最优/最坏tie界，同CR用最小整tie集合、不能GT拆同分。只允许预定redc候选采用，不事后改选linear：总体同数/同CR严格优于simple、score、linear，各组FA不增，同CR最长FR不增；不能精确匹配CR的整tie不确定项保守记不能确认通过。FR与ED/MISSING分开。失败封存不加预算、不读TEST；通过后才能冻结报告已暴露TEST，不据TEST选参。

实现入口`crane_project/tools/run_port_reliability_redc_size_v1.py`；公式/线上decision在`utils/port_reliability_redc_size_v1.py`，Torch头/梯度在`utils/port_reliability_redc_size_v1_torch.py`。独立source/contract SHA清单封存上述设定及全部依赖。所有产物放`work_dirs/port_reliability_redc_size_v1/RUN_ID`，分析包仅一个在work_dirs根目录；分析/核验/记录同步后仅清理本次两端包，完整结果、权重、缓存保持。

### 34.2 执行记录（已完成运行）

初始预检：本地18项数值/接口检查通过，8项Torch检查因本地缺Torch未执行，随后在服务器真实Torch环境全部执行通过；Python编译、shell语法和来源清单检查通过。服务器核实路径`/media/omnisky/personal_files/ljj/symEOOD`、origin`https://github.com/lusirs123/mmrotate-crane1.git`，两端起始main96ca7dd。服务器既有工作区/index指纹b48cc0…/d54105…前后保持；本地已有7份删除文档和未跟踪检索词文件不纳入本次提交。初检GPU2/3均空闲，实际使用GPU2。最终性能结论见34.3。

实际已在GPU2完成的运行入口（存档，无需补跑）：
```bash
cd /media/omnisky/personal_files/ljj/symEOOD
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
git pull --ff-only
CUDA_VISIBLE_DEVICES=2 bash tools/run_port_reliability_redc_size_v1.sh 20261008_redc_v1
```


### 34.3 实测、独立复核与决定

**执行事实。** 新源码021df7b在本地/服务器main同步。Git初次push审批因远端归属不明拒绝；只读gh核验登录lusirs123、仓库lusirs123/mmrotate-crane1的ADMIN权限、PUBLIC可见性与默认main后，同一动作获准推送。服务器原127.0.0.1:17890代理未连接，通过临时SSH loopback17890→本机Clash7890恢复git pull，无持久代理修改。第一次临时转发审批超时未执行，按允许重试一次后成功。服务器既有工作区/index指纹b48cc0…/d54105…保持。用户既有本地删除文件未纳入本次提交。

服务器Python3.8.20、Torch1.13.1+cu117、NumPy1.24.4、物理GPU2 GTX1080。26项服务器测试全部通过，无跳过；独立两步smoke丢弃。完整TRAIN2558输出、71错误（Real70/Sim1）；两臂各100轮×10批=1000更新，总2000。唯一末态epoch100，没有VAL选择epoch、无追加训练。完整VAL887帧/886输出、663正确/223错误，1漏检不进入FA/FR。实际冻结M全六字段重放差均0，GT TRAIN周期角舍入最大1.78814e-7rad保持原标签；无B/几何头更新或新检测推理。Torch/NumPy最终risk最大差linear1.70e-7、redc2.44e-7，完整原框、score、中心/方向标志保持。

**预定工作点（不同方法门限不可直接当等接受数比较）。** 各域/视频正确保留>=95%的单一全局VAL门限：redc risk<=0.8828651140369607，linear<=0.7000301508590092。redc接受861、FA201/FR3/ED22/CR660，接受风险23.3449%，正确保留660/663=99.5475%；seq07正确保留60/63=95.2381%。linear接受857、FA206/FR12/ED17/CR651；simple同要求的全局点为856接受、FA200/FR7/ED23/CR656；score861接受、FA201/FR3/ED22/CR660。redc与score在这个点的逐帧接受集合完全相同，不称“新增检出收益”。原正式simple仍843接受、FA190/FR10/ED33/CR653，接受数/正确保留不同，不能直接用201与190比较判优劣。

| 完整VAL同861接受 | FA | FR | ED | CR |
|---|---:|---:|---:|---:|
| redc候选 | 201 | 3 | 22 | 660 |
| 同输入linear控制 | 208 | 10 | 15 | 653 |
| 冻结simple评分 | 202 | 4 | 21 | 659 |
| score | 201 | 3 | 22 | 660 |

| 同正确保留参照 | 候选FA | simple FA | score FA | linear FA |
|---|---:|---:|---:|---:|
| 全VAL CR660 | 201 | 202 | 201 | 215 |
| Real CR196 | 157 | 156 | 157 | 167 |
| Sim CR464 | 44 | 45 | 44 | 48 |
| seq07 CR60 | 144 | 130 | 144 | 147 |
| seq14 CR136 | 13 | 13 | 13 | 13 |

同CR是每个报告组分别取控制的最小整tie集合，仅用于离线公平排序，不是重新定义线上门限或按视频部署。seq07同204接受数时candidateFA144、simple141、score144、linear145；同CR60更清楚地显示对simple多14FA。各组最长FR均满足本轮同CR保护，没有把连续ED算为FR。5项失败为：总体同数/同CR对score严格增益未成立、Real同CR相对simple退化、seq07同数/同CR相对simple退化；它们不是5个独立错误事件。没有保留率、连续FR或工程失败，不能再把未采用原因写成“评价把拒错误算为误拒”。

错误AUROC：全VAL redc0.667519、linear0.608607、simple0.785626、score0.684394；Real redc0.654673，seq07 redc0.471389，Sim redc0.696929。TRAIN类平衡末态loss从score初始化0.913548降至redc0.475896、linear0.370829；TRAIN错误AUROC分别0.850864/0.918698。loss下降、优于新linear控制不等于优于已保留simple/score，也没有稳定跨视频收益。第35节复核更正了simple/score的文字末位抄录，原报告/模型/逐帧产物不变。

**机制事实与推断。** 学到β=-0.476239。TRAIN2049/2558、VAL749/886输出的第二个ReLU隐层8个激活全部为0，因此它们的温度完全相同0.778454；该子集的risk是score统一单调变换，不能通过内部特征改变子集内部排序。其余样本仍有不同温度，不能把整个网络说成完全没用特征。smoke第二步首层grad_norm0.05117非零，说明图像特征路径连接成立；大量末态零激活是本版本表达/拟合的限制线索，不是已经证明唯一根因或整个ReDC无效。不追加epoch、改激活/容量/特征/权重或另选linear。

**覆盖及核验。** 中心输出帧命中874/886=98.6456%，输出覆盖886/887=99.8873%，全帧中心正确覆盖874/887=98.5344%，拒尺寸不删中心。完整TRAIN/VAL3445行独立NumPy网络重放、210组四状态/同数/同CR参照、120个成对AUROC、全部门限及2000更新/批次次序复算一致，risk最大差1.03e-15。审查器新增7项数值测试通过；没有重新拟合或改变阈值。

**决定。** status=VAL_FAILED_STOP，未通过预先合同，不采用本次ReDC启发版本，原M/simple/policy保持。完整VAL已完成，没有TEST访问、选权或门限调整；失败仅写本运行记录，不写主要结果或论文新成果。本结果限定于本项目本次特征/目标/结构/预算，不构成对ReDC原论文的复现失败结论。

产物统一位于`work_dirs/port_reliability_redc_size_v1/20261008_redc_v1/result`；包含models.json/final_heads.pth、train/val_features.npz、逐帧scored_TRAIN/VAL、train_log、smoke/report/cutoffs/completion及local_review。唯一回传包`work_dirs/port_reliability_redc_size_v1_20261008_redc_v1.tar.gz` SHA07a027d4a1122f5af03b02ae4ef835c7d2e4ffe0faad8ffb6ad377245261ceef，路径/类型和完整产物SHA核验通过。report SHAe18466a22f053501db264a8201421c0ffdcdcb519ac5978bf287b91e3ef981f8，models SHA1e648ace3f2f350c284dc27a61f8a356f366f95a088b8569900492a71c1c4417。

**同步与清理完成。** 审查器及分析提交425654c推送、服务器快进拉取；新增7项审查测试在两端均通过，服务器3445行/210组/120个AUROC及2000更新复核一致，风险重放最大差5.00e-16，报告与模型SHA和本地相同。两端审查末位数值因NumPy/BLAS不同略有差异，不要求审查JSON逐字节相同。服务器git diff --binary与暂存区指纹仍分别为b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1、d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e；既有改动保持。分析同步后，两端仅删除上述SHA核验的本次tar.gz，完整结果目录、权重及原缓存全部保留，不清理其他归档。

## 35. 温度／有符号质量修正的单项有限对照（2026-10-08）

### 35.1 已授权设计与预先合同

用户授权按评分公式限制的建议做一次有限检验，沿用本地修改、Git同步到`/media/omnisky/personal_files/ljj/symEOOD`、服务器后两卡执行、回传核验记录及仅清理本次两端压缩包的流程。第34节模型未采用，原M、simple/policy、中心/方向及输出身份固定；不恢复修框/深度，不读TEST，不自动追加版本。

**新增事实。** 第34节末态第二隐层全零区域，TRAIN包含7/71个错误，VAL包含207/223个错误，seq07包含157/162个错误；不是仅因全部网络没有梯度。温度风险logit为`-(z/T(h)+beta)`且T>0、beta全局，任何score>.5项的错误风险都低于任何score<.5项；无法重新排序这类错误—正确配对。VAL138个错误score>.5、107个正确score<.5，seq07为94/29。这是数学约束与当前离线事实，不认定唯一根因；本公式对所有低score错误／高score正确的配对同时有有利的固定次序，故解除约束也不保证总体收益。

**唯一增量。** 复用第34节SHA核验的实际冻结stem特征288维及原M三描述量，不再检测/提取、不改变样本/激活/归一化/类权重。两臂同290→16→8→1 ReLU与全局beta，每臂4802参数；temperature=`-(z/(.25+softplus(a(h)))+beta)`，residual=`-z+a(h)+beta`。最后权重均0、beta均0；中性末层bias分别inverse-softplus(.75)/0，均在1e-5 logit容差内还原score风险（residual为精确零修正）。同隐层初值，保留旧linear RNG消耗以重现第34节temperature初值；参数/优化器独立。非完整ReDC复现，不将类平衡风险称为真实错误概率。

完整TRAIN2558/71错误，两臂各100轮、batch256、各1000更新，总2000更新，seed1701同批次序；AdamW lr.001、仅weights衰减1e-4、clip5。独立2步smoke丢弃，检查首层后续梯度、Torch/NumPy重放、输入detachment；只用末态100轮。temperature末态参数必须重现第34节max差<=1e-6、批次digest相同，否则工程失败停止，不判候选优劣。完整TRAIN拟合后才打开VAL特征用于评价，不增加fit/probe等角色。

**验收分开。** 机制诊断：分域/逐视频报告高score错误—低score正确配对排序、第二隐层全零的正确/错误组成，TRAIN/VAL均报告；只是解释，不作选权或替代验收。采用门槛：仍在VAL使用单一全局整tie门限满足各域/视频正确保留>=95%，同接受数/同CR总体FA严格优于temperature、冻结simple及score，每组FA不增、同CR连续FR不增；ties不可精确匹配保守记未确认。原policy固定点另报，FA/FR/ED/CR/MISSING及中心输出帧命中、输出覆盖、全帧正确覆盖保持原口径。VAL是反复使用的开发数据，TEST多次暴露；本轮无论VAL是否通过均不访问TEST，不据其选模型/门限。

实现入口`crane_project/tools/run_port_reliability_readout_compare_v1.py`，独立审查器`review_port_reliability_readout_compare_v1.py`；源文件及合同SHA封存，既有第34节代码/权重/结果不覆盖。结果集中`work_dirs/port_reliability_readout_compare_v1/RUN_ID/result`，已有特征直接引用原路径，不另拷ROI/cache。仅本次一个回传包在work_dirs根目录，核验分析同步后只删两端这个包；未采用实验仅记本运行记录。

### 35.2 执行进度

初始本地15项数值/来源/接口测试通过，7项Torch测试因本地无Torch随后交服务器执行；SHA、Python3.8与shell语法核验通过。服务器只读检查main77cc965、原工作区/暂存SHA保持；执行前GPU2有其他任务，GPU3空闲，脚本只允许在2/3中选空闲卡。初始预检不构成性能结论，最终实测如下。

### 35.3 实测、独立核验及决定

**执行事实。** 源码9bc0136已推送并在指定服务器main快进拉取，保留既有改动；临时loopback转发仅用于Git拉取，没有持久代理修改。Python3.8.20/Torch1.13.1+cu117/NumPy1.24.4、后两卡空闲GTX1080执行，22项服务器测试全通过，两臂独立2步smoke丢弃。完整TRAIN2558、完整VAL887/886输出、663正确/223错误、1漏检。各100轮/1000更新，总2000；temperature末态全部参数与第34节max差0、批次digest相同。复用原冻结特征，无新检测或几何头forward，无新角色/扰动/重采样/选权/追加预算。

**机制检验成立但采用收益不成立。** 高score错误—低score正确的配对，temperature在TRAIN15000、VAL14766对中正确排序均0；residual分别14921/15000=99.4733%、1128/14766=7.6392%。配对复用帧、彼此相关，不是独立事件样本。seq07仅122/2726=4.4754%。residual已能跨score=.5分界重新排序，所以不是代码未解除数学限制；但该能力主要在TRAIN，未迁移到VAL。

| 排序／拟合指标 | temperature | residual |
|---|---:|---:|
| TRAIN类平衡末态loss | 0.475896 | 0.121823 |
| TRAIN错误AUROC | 0.850864 | 0.996506 |
| 全VAL错误AUROC | 0.667519 | 0.599267 |
| Real VAL错误AUROC | 0.654673 | 0.626676 |
| Sim VAL错误AUROC | 0.696929 | 0.636629 |
| seq07错误AUROC | 0.471389 | 0.455124 |

这表现为更强TRAIN拟合而VAL排序退化，不能用loss/训练AUROC或解除公式限制宣布可靠性提升。simple全VAL AUROC0.785626、score0.684394。

**预定全局使用点。** residual门限risk<=0.9855444386015616，859接受、FA219/FR23/ED4/CR640/MISSING1；正确保留640/663=96.5309%，Sim441/464=95.0431%，每域/视频>=95%满足。temperature仍861接受、FA201/FR3/ED22/CR660。接受数不同，不直接把219与201作为公平排序比较。原policy固定点未改，FA190/FR10/ED33/CR653，843接受。

| 全VAL同859接受 | FA | FR | ED | CR |
|---|---:|---:|---:|---:|
| residual候选 | 219 | 23 | 4 | 640 |
| temperature控制 | 200 | 4 | 23 | 659 |
| 冻结simple评分 | 202 | 6 | 21 | 657 |
| score | 200 | 4 | 23 | 659 |

| 同正确保留参照 | residual FA | temperature FA | simple FA | score FA |
|---|---:|---:|---:|---:|
| 全VAL CR640 | 219 | 182 | 184 | 185 |
| Real CR199 | 173 | 164 | 161 | 163 |
| Sim CR441 | 46 | 38 | 33 | 37 |
| seq07 CR63 | 160 | 151 | 141 | 150 |
| seq14 CR136 | 13 | 13 | 13 | 13 |

同CR参照只作离线公平排序，不按视频改线上门限。residual全部23个FR均在Sim，最长连续FR4；同全VAL CR640控制最长FR均2。Real仅检出2/175错误、seq07仅检出2/162错误，正确Real全部保留；Sim检出2/48错误，却误拒23/464正确。37项失败含同指标/域与视频重复关系，不是37个独立失败帧；本轮没有95%正确保留或工程失败，排序和连续FR保护实际退化。

**新的限制线索。** residual第二隐层全零TRAIN1910/2558包含1/71错误，VAL610/886包含173/223错误，seq07包含137/162错误；零区域数量虽少于temperature，绝大多数VAL错误仍落入原score加常量的区域。Sim正确框的95%所需门限为0.985544，比Real的0.639984明显高，单一全局门限被Sim决定，从而几乎全收Real错误。这是跨域风险值失配的事实；同时两域各自AUROC也退化，故不能把失败全部归于全局门限，也不改成分视频/分域门限追求通过。根因是否来自冻结特征、训练错误支持、监督/拟合方式尚未隔离，不宣布ReLU或数据缺乏为唯一根因。

**工程与独立核验。** 首层smoke第二步梯度temperature0.05117、residual0.05893非零；输入detach、clip、参数数目/隐层初值、NumPy/Torch重放通过，末态风险最大差2.44e-7/6.44e-7。3445行两端独立CPU网络重放、210组四状态/同数/同CR、120组pairwise AUROC、全部门限、2000更新/批次digest及机制诊断均一致；本地重放最大差1.89e-15，服务器0。`local_review.json`为服务器审查，`mac_review.json`为本地审查，末位浮点可不同但结论/原产物SHA保持。原M/score/输出及中心/方向标志保持；中心874/886=98.6456%，输出覆盖886/887=99.8873%，全帧中心正确覆盖874/887=98.5344%。

**决定。** VAL_FAILED_STOP，不采用residual或第34节temperature，不进入TEST。结论限定为“解除正温度的排序限制不足以解决本项目当前冻结特征质量映射的泛化”，不是全部可靠性方法无效或原ReDC复现失败。不自动追加epoch、调激活/容量/特征/类权重/分域门限；正式M/simple继续保持，失败只记本运行记录，不写论文主要成果。

产物统一`work_dirs/port_reliability_readout_compare_v1/20261008_readout_v1/result`，原特征引用第34节，未拷旧缓存。唯一包`work_dirs/port_reliability_readout_compare_v1_20261008_readout_v1.tar.gz` SHA4fcf151ca6f967a145d73c76aec1d227b7b42f07285a49b3fc673dc7b508c53c，路径/类型/完整回执核验通过；report SHA382f5c94ea636c1ca1a4ea6c0f63a57122f950f6be03de10f3e123e6a510a673，models SHAb49cc7c69ba8285cfc6f137c56b992bdc157410c070bbdbe0e7eb736922f41aa。服务器工作区/暂存SHA前后仍b48cc0…/d54105…，退出码0。

**分析同步与清理已完成。** 分析提交bd93336推送并在服务器快进拉取；本地`mac_review.json`同步到服务器同目录，SHA62ed817e4365c55a8cac7907d03314552e0f051912d4d54f2fccf2cbbed695b2，两端一致。确认记录已同步、完整结果与权重存在且原工作区/暂存SHA保持后，两端仅删除上述本次tar.gz；全部结果、权重、原特征及其他归档保留。没有停止其他作业或改正式policy。


## 36. 第35节两臂的冻结TEST补评（2026-10-09，已执行并复核）

### 36.1 授权与证据身份

用户在第35节失败后明确要求补齐TEST，以了解现有两臂在不同分布上的固定表现。最初授权仅限本地修改与检查，服务器暂不可连接；之后用户明确确认服务器恢复并授权在原项目/环境执行，实测见36.5。因此这是独立的**冻结TEST诊断入口**，不是改写第35节合同：历史VAL_FAILED_STOP、未采用、当时未读TEST的事实保持；最初本地实现阶段未运行实际TEST、未提交/推送Git、未连接服务器。TEST已多次暴露，不能称未接触的独立测试，更不能依据此次表现重新拟合、调整门限、挑选权重/结构或自动晋级。

两臂固定第35节最终epoch100，各1000更新；`models.json` SHA `b49cc7c69ba8285cfc6f137c56b992bdc157410c070bbdbe0e7eb736922f41aa`，标准化仍为完整TRAIN拟合所得。四种评分使用原VAL单一全局risk_le：temperature `0.8828651140369607`、residual `0.9855444386015616`、full_simple `0.43131551898452714`、score_only `0.7688139975070953`。这些门限在TEST上的正确保留并不保证达到95%，需如实报告实际值。原正式simple policy使用点单独报告，不能与VAL95参照混为一谈。

### 36.2 实现与保护

新增独立入口 `crane_project/tools/eval_port_reliability_readout_test_v1.py`（check/evaluate）、独立CPU审查器 `review_port_reliability_readout_test_v1.py`、两个同名utils、固定protocol/sources、单测和 `tools/run_port_reliability_readout_test_v1.sh`。旧TRAIN/VAL入口、拒绝TEST的连续性统计限制、来源清单、模型、门限和结果均未修改。TEST统计实现沿用原四状态与断帧重置口径，单独验证TEST身份；合成样例已确认与旧统计合同一致。

引用原1440帧港口TEST（real_seq03 200、real_seq04 668、sim_seq09 572）和历史GT-free ROI收集：`work_dirs/port_geometry_size_boundary_v1/size_candidate_downstream_20261007_191454_1852169/collect/port_test/roi.pt`，SHA `a0a1cb1f4245238b90f409f5e8a6ed5b306f8ae07dbfb06f1c8377891371003d`。仅使用该收集中的B/M、ROI及实际sx/sy，不加载其尺寸候选权重、不重启修框或深度优化。冻结σ1.5/ep03几何头提取实际32×9×9 stem，3×3平均池化288维，加交付M描述符3维。重放头输出只做来源核对，不替换已交付框。

输入逐文件SHA、图像/帧/坐标/score/输出身份、冻结头状态及原simple三标志均核对；线上传入的字段白名单排除GT/标签/离线误差/域/视频。只允许尺寸risk与size_accept变化，中心、方向、框、score和漏检保持。无框时不产生质量特征或判断，MISSING单列。结果拒绝覆盖旧目录，包括拒绝覆盖时也不往历史目录写failure文件。

完整TEST分域/逐视频报告固定门限FA/FR/ED/CR、接受覆盖、正确保留及全帧正确覆盖、接受后错误比例、错误AUROC及连续FR；另报同接受数score/simple/另一臂对照、整tie同CR参照与同数tie上下界。上述匹配只为离线比较，不生成TEST部署门限；连续拒绝错误与连续误拒正确分开。中心分别报告输出帧命中率、输出覆盖率及全帧中心正确覆盖。独立审查器重放JSON网络、逐帧风险/三标志、四状态、同数/同CR、tie上下界和AUROC。

### 36.3 本地检查与待运行项

**事实：** 新增21项测试中18项通过、3项Torch测试因本地无Torch跳过；原第35节回归22项中15项数值/来源/接口通过，7项Torch在本地跳过（其历史服务器通过记录仍见35.3）。新增Python3.8语法/编译与shell语法检查通过；新旧来源闭包、末态模型/门限及已有文本证据SHA核验通过。仅以合成数据验证新评分/评价/独立审查的逻辑，没有对真实TEST计算候选分数或指标。

本地未保存上述TEST ROI与原σ1.5/ep03权重，check模式明确列为缺失；evaluate必须找到并核对原SHA，否则停止，不自动生成新缓存，也不从记录推断服务器文件仍然存在。真实Torch特征重放、服务器环境及完整TEST表现均待运行，因此目前没有新性能收益或失败结论。

### 36.4 初始预留服务器入口（随后已按36.5执行）

源码目前未提交/推送。后续通过Git同步本轮新增文件后，在服务器原项目目录及mmrotljj环境运行以下单行入口：

```bash
bash tools/run_port_reliability_readout_test_v1.sh 20261009_frozen_test_v1
```

脚本默认仅在物理GPU2/3选择空闲卡；若两卡忙则停止，显式CUDA_VISIBLE_DEVICES可指定卡。自动执行单测、静态check、冻结evaluate和独立review，不训练、不改门限、不连接其他目录。一个运行的日志与结果集中于 `work_dirs/port_reliability_readout_test_v1/20261009_frozen_test_v1/`；退出时生成唯一 `work_dirs/port_reliability_readout_test_v1_20261009_frozen_test_v1.tar.gz`，包含结果/特征/日志，不另复制旧ROI或权重。原ROI或权重缺失时应先解决原产物可用性，不能重跑旧候选替代。本轮未生成、传回或删除压缩包。


### 36.5 服务器冻结TEST实测、回传核验与决定

**执行事实。** 用户确认服务器恢复并授权运行。源码 `0ee2256` 推送原origin `lusirs123/mmrotate-crane1`，在 `/media/omnisky/personal_files/ljj/symEOOD` 的main快进拉取；环境为mmrotljj/Python3.8.20/Torch1.13.1+cu117/NumPy1.24.4。原代理127.0.0.1:17890不可用，本次拉取临时清空代理配置/环境成功，不改持久设置。物理GPU2空闲并实际执行，21项服务器测试全通过，原ROI/权重SHA通过；无需新增检测推理、训练或几何修改。1440帧、1431输出、1089正确尺寸/342错误尺寸/9漏检，最终状态 `EXPOSED_TEST_DIAGNOSTIC_COMPLETE_KEEP_VAL_FAILURE`，退出码0；候选与门限固定，未调用TEST拟合、采用门槛或选权。

**固定原VAL门限实际点（不是同接受数比较）：**

| 方法／来源 | 接受数 | FA | FR | ED | CR | 正确保留 | 接受后错误比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| temperature | 1424 | 338 | 3 | 4 | 1086 | 99.7245% | 23.7360% |
| residual | 1354 | 283 | 18 | 59 | 1071 | 98.3471% | 20.9010% |
| full_simple评分／VAL95门限 | 1360 | 287 | 16 | 55 | 1073 | 98.5308% | 21.1029% |
| score／VAL95门限 | 1402 | 317 | 4 | 25 | 1085 | 99.6327% | 22.6106% |
| 正式原simple policy（未改） | 1330 | 275 | 34 | 67 | 1055 | 96.8779% | 20.6767% |

residual相对正式policy多接受24帧，其中多保留16正确、也多接受8错误；不能只比较FA283与275来判排序优劣，也不能只比较FR18与34宣布全面提升。residual接受覆盖1354/1440=94.0278%，全帧正确覆盖1071/1440=74.3750%。Real固定点FA262/FR5/ED58/CR534，Sim FA21/FR13/ED1/CR537；Real接受后错误比例32.9146%。各视频正确保留均>=95%，这只是该冻结TEST点实测，不是泛化保证。

**公平排序比较：** 全TEST同1354接受，residual FA283/FR18，temperature FA320/FR55，simple FA284/FR19，score FA304/FR39；相对score少21个FA，相对simple仅少1个。全TEST同CR1071，residual FA283、temperature331、simple285、score311，均可整tie精确匹配；但候选连续FR最长3、temperature1、simple/score2，仍没有保护连续误拒。全TEST错误AUROC：residual0.600513、temperature0.587504、simple0.766600、score0.591736，不能从一个高覆盖点宣布总体排序优于simple。

| 按组同候选接受数 | 接受数 | residual FA／FR | simple FA／FR | score FA／FR | temperature FA／FR |
|---|---:|---:|---:|---:|---:|
| Real总体 | 796 | 262／5 | 269／12 | 282／25 | 294／37 |
| Sim总体／seq09 | 558 | 21／13 | 22／14 | 21／13 | 20／12 |
| Real seq03 | 141 | 52／2 | 52／2 | 72／22 | 74／24 |
| Real seq04 | 655 | 210／3 | 211／4 | 209／2 | 212／5 |

分组匹配不等于把各视频分别定线上门限；各组参照也不能直接相加为总体同数参照。seq03比score改善明显，但与simple相同；seq04比score多1个FA；Sim比temperature多1个FA且误拒连续段更长，收益跨视频/域不一致。Sim residual AUROC0.457934，temperature0.599504、simple0.535950、score0.586281，固定点仅检出1/22错误却误拒13/550正确。因此既不能说TEST毫无信号，也不能用Real局部信号掩盖Sim退化，或把它追溯为原VAL合同通过。

**已观察到的机制限制。** residual TRAIN错误AUROC0.996506、VAL0.599267、TEST0.600513，强拟合未变成稳定泛化；TEST跨score=.5的“高score错误—低score正确”配对8213/44000=18.6659%，比VAL7.6392%高，但远低于TRAIN99.4733%；Sim为0/1212。配对共享帧，非独立事件，不能用对数当独立样本。TEST第二隐层全零873/1431，其中184/342错误；seq04全零包含137/218错误。该区域残差输出只能是固定常量，风险退化为score logit加常量；这是限制事实，不证明ReLU、冻结特征或TRAIN支持中的任一项是唯一根因。

高FA也受覆盖工作点的数学限制：1354接受、可用正确框1089，同数完美排序也至少FA265，当前283；单靠此接受数量下的排序最多再降低18个FA。此界不说明判断器已经理想，而是说明高覆盖和降低错误使用之间确有取舍，不能承诺只换判断器就让该点FA大幅下降。

**保护、回传与独立审查。** 原M重放6项最大差全部0；中心1430/1431=99.9301%，输出覆盖1431/1440=99.3750%，全帧中心正确覆盖1430/1440=99.3056%；框、score、输出/漏检、中心与方向标志保持。服务器与本地逐帧网络/四状态/同数/同CR/tie/AUROC复核通过，共1440行/1431输出、108组状态匹配、48组AUROC；本地风险最大差1.17e−15、服务器0。补充离线机制分析12组也独立复核，仅解释固定结果，不用于选择。

结果统一在 `work_dirs/port_reliability_readout_test_v1/20261009_frozen_test_v1/result`；服务器 `independent_review.json` 与本地 `mac_review.json` 分开保留，两端共享 `diagnostic_analysis.json`、`analysis_independent_review.json`。唯一回传包 `work_dirs/port_reliability_readout_test_v1_20261009_frozen_test_v1.tar.gz`，2284666字节、26成员，SHA `34d053eedfd41e7a5ba80cb02e1326e3c83343fb9a3f639c8c26528914539626`；路径/类型/原产物回执全部核验。report SHA `1a432a20042d94f4439e19338d10519d62128a8b310647750aa311fb5072b2bd`，模型/门限/原policy字节不变；服务器工作区与暂存指纹前后仍为 `b48cc0…`／`d54105…`。

**决定与建议。** 保留正式M/simple/policy，两臂仍不采用，原VAL失败不撤销。此次TEST补评证明跨score修正有局部作用，但没有形成超过simple的稳定跨域、逐视频及连续性优势；不是原ReDC论文无效的结论。建议收束当前两臂，不追加轮数/调门限/换激活或按TEST改选；可继续整理已有正式检测与三分量可靠性成果及这些负结果的适用边界。失败候选只记运行记录，不写成论文正式提升。完整结果保留，压缩包在分析记录同步与产物复核后按既有约定仅清理本次两端包，清理回执另记。

**分析同步与清理完成。** 分析提交 `52ba255` 已推送并在服务器快进拉取；同步时GitHub直连超时，临时SSH仅绑定服务器回环17890转发至本机Clash7890完成拉取，连接结束即撤销，不改持久配置。两端report/mac_review/诊断分析与复核SHA一致。原完整结果、两臂模型、门限、原policy、TEST ROI与σ1.5/ep03权重核验保留后，仅删除本次两端 `port_reliability_readout_test_v1_20261009_frozen_test_v1.tar.gz`，其他文件与归档未删除。服务器清理回执已回传至同一RUN_ID的 `archive_cleanup.json`；原工作区/暂存指纹保持。


## 37. 固定特征来源对照（2026-10-09，已完成；不采用）

用户授权按已同意设计完成本地实现、必要检查、Git同步、原服务器工作目录运行、回传独立核验、结果记录及仅清理本次压缩包。本项只优化尺寸可靠性，不修框、不扩展深度/几何路线；正式M、simple、原policy与历史实验身份保持。

**唯一处理是特征来源。** A为原midpoint stem32×9×9的3×3平均池化288维；B为B原生Top-1实际选中层/位置/anchor处、进入retina_reg的256×3×3局部特征2304维，零填充与真实卷积一致。通过只读Python返回事件追踪原 `_get_bboxes_single` 的实际 `topk_inds`，不改变原方法、不重新topk、不做最近框匹配；另外重放局部回归卷积、原B框及冻结state，记录索引、tie、坐标scale与patchSHA。B的FPN被分类/回归共享，没有独立回归塔；监督目标是最终M尺寸质量，B特征未必解释midpoint额外误差。

两臂分别只以完整TRAIN拟合无标签中心化PCA，固定256维、不白化、不搜维数，再各以TRAIN均值/标准差规范化。共同M相对尺寸/比例2维+PCA256维进入258→16→8→1 ReLU网络，另接score logit与beta；各4290参数，同seed1701初始化，末层与beta为0，风险为sigmoid(-z+a(h)+beta)。因此A必须重新训练，旧4802参数版本只作历史结果，不冒充公平控制。

完整TRAIN2558输出、71个真实尺寸错误；标签为最终M与GT规范长短边最大相对误差>10%。相同100epochs/1000updates、batch256、seed1701顺序、N/(2Nclass)平衡BCE、AdamW .001、仅weight decay1e-4、clip5。两步smoke丢弃，正式重新初始化，固定epoch100，不选权重/增预算/改激活。

完整TRAIN拟合→完整VAL887开发/门限→候选冻结后TEST报告；不新增fit/probe角色。VAL只选一个全局整tie最小门限，保护每域/每视频正确保留>=95%。仅native可晋级：总体同接受数、同CR均严格少于重新训练的midpoint、冻结simple与score；逐组FA不增、同CR连续FR最长不增、tie保守处理。失败停止，不换超参再试，不读TEST；TEST反复暴露，永不用于选择。原policy点单列，中心输出帧正确率/输出覆盖/全帧正确覆盖分别报告。

新入口 `tools/run_port_reliability_feature_source_v1.sh`；数据/模型/审查/日志统一在 `work_dirs/port_reliability_feature_source_v1/RUN_ID/`，唯一结果回传包在work_dirs根目录。实现/协议/来源SHA为新文件，不修改旧实现与来源清单。已完成本地数值检查；Torch在本地缺失，对应检查留给服务器实际执行。初始实现时尚无GPU训练或性能收益结论；后续实测见37.1。

**审查器工程修正（运行权重/合同不变）。** 静态复核发现native原始NPZ为float32，而PCA在float64拟合；审查中的均值也应先转float64，不能把float32求和舍入差当拟合错误。审查器已修正并改为逐项核对report记录的原Git提交/源码闭包，保持原运行来源身份。训练运行在提交 `fa1ab31`，不因审查器修正重跑训练、改参数/门限或重写report；修正版审查将单独执行。

**审查耗时定位。** 已完成TRAIN/VAL后，只结束了本次旧审查进程（核对PID命令与cwd后SIGINT，未中断训练/更改结果）。栈显示逐帧 `native[features]` 会重复解压整个NPZ成员；修正为一次载入后逐行核对patchSHA。连同float64均值修正单独重跑独立审查，原脚本退出状态/旧日志保留，不能将其称为模型训练失败或额外训练。


### 37.1 完整TRAIN/VAL实测、回传独立核验与决定

**运行身份与合同。** 服务器原目录 `/media/omnisky/personal_files/ljj/symEOOD`、mmrotljj、物理GPU2（GTX1080），运行源码提交 `fa1ab31`。服务器15项测试全通过；本地11项数值检查通过、4项Torch缺失跳过，Torch实际由服务器执行。完整TRAIN2558帧/71尺寸错误，两臂各100epoch/1000updates、相同顺序、4290参数；两步smoke丢弃、正式重新初始化，中性初始logit精确为-z（沿用原score logit的数值截断约定），不宣称浮点风险与1-score逐位相等。PCA只拟合TRAIN，固定256维，A保留方差100%、native92.1618%；PCA阶段不白化，后续逐维TRAIN标准化会重标度主成分，两步要一起描述。原B完整TRAIN及VAL原图框/score重放6项最大差全部0；状态digest保持；M与原标志/标签按原证据保留，不修框。

**总体公平比较，完整VAL887帧、886输出、663正确/223错误/1缺失。** 下表第一列为同836个接受数的离线排序参照，最后一列为各方法整tie最小门限保留同653正确的离线参照；均不作为新的在线分域/分视频门限。

| 方法 | 同836接受FA | FR | 同CR653的FA |
|---|---:|---:|---:|
| native候选 | 183 | 10 | 183 |
| 同PCA/同容量midpoint控制 | 189 | 16 | 195 |
| 冻结simple评分 | 188 | 15 | 190 |
| score | 188 | 15 | 197 |

候选实际固定VAL点为FA183/FR10/ED40/CR653/MISSING1，接受覆盖836/887=94.2508%，正确保留653/663=98.4917%，全帧正确尺寸覆盖653/887=73.6189%，接受后错误比例183/836=21.8900%。正式原simple policy仍为FA190/FR10/ED33/CR653、843接受（22.5386%错误比例）；同正确保留下本候选少7FA，有局部信号。原policy字节/门限不变；A自身VAL门限实际为866接受、FA206/FR3/ED17/CR660，不能直接把其FA206与候选183当作同覆盖排序比较。

**分域/逐视频。** 原始固定候选点如下（缺失单列、不进入四状态）；各组正确保留均>=95%。

| 组 | 帧/输出 | FA | FR | ED | CR | 缺失 | 接受覆盖（全帧） | 正确保留 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| domain:real | 375/374 | 141 | 3 | 34 | 196 | 1 | 89.8667% | 98.4925% |
| domain:sim | 512/512 | 42 | 7 | 6 | 457 | 0 | 97.4609% | 98.4914% |
| sequence:real_seq07 | 226/225 | 128 | 3 | 34 | 60 | 1 | 83.1858% | 95.2381% |
| sequence:real_seq14 | 149/149 | 13 | 0 | 0 | 136 | 0 | 100.0000% | 100.0000% |
| sequence:sim_seq10 | 512/512 | 42 | 7 | 6 | 457 | 0 | 97.4609% | 98.4914% |

| 组 | 候选接受数 | native FA | 同数midpoint FA | 同数simple FA | 同数score FA | 同CR native/simple/score FA |
|---|---:|---:|---:|---:|---:|---|
| domain:real | 337 | 141 | 147 | 148 | 146 | 141/156/157 |
| domain:sim | 499 | 42 | 46 | 40 | 42 | 42/35/42 |
| sequence:real_seq07 | 188 | 128 | 134 | 130 | 132 | 128/130/144 |
| sequence:real_seq14 | 149 | 13 | 13 | 13 | 13 | 13/13/13 |
| sequence:sim_seq10 | 499 | 42 | 46 | 40 | 42 | 42/35/42 |

组内参照各自匹配，不能相加冒充总体同数参照。关键失败是Sim同499接受候选FA42而simple40；同CR457为候选42、simple35，候选多7FA。seq14同CR136为候选FA13、midpoint12。合同5项失败包含Sim在domain/sequence的重复记账，并非5个独立问题；总体严格FA增量、95%正确保留及同CR连续FR保护均通过，候选最长连续FR为1，但逐组FA不增未通过。因此不采用，不进入TEST，不追加epoch、调激活/维数/门限，不用已经暴露TEST重新挑方案。

**拟合、排序与机制线索分开。** 两臂TRAIN错误AUROC均1.0000，最终平衡BCE为midpoint0.066999、native0.065175（共同初始0.913548）。VAL总体AUROC为native0.716610、midpoint0.734648、simple0.785626、score0.684394；native并非整体排序最强。seq07为native0.543798、midpoint0.463159、simple0.539193，只有有限增量，不能说高FA已解决。

native第二隐层全零：TRAIN2142/2558、错误0/71；VAL701/886、错误157/223。midpoint控制VAL全零476/886、错误63/223。全零区最终残差只剩末层bias+beta，风险退化为-score_logit+常量，无法用该隐层信息重排其中错误；这是明确限制，不证明ReLU、PCA、FPN或TRAIN支持任一项是唯一根因。B特征与M目标还存在范围差距：VAL有48个B正确而M错误，native仍接受43；但175个B/M均错误也接受140，所以不能把失败全部解释成midpoint新增误差。统计单位仍是相关视频帧，不能当独立事件数。

**中心三口径及保护。** 中心命中按输出帧统计，另报输出覆盖与全帧正确覆盖；仅继承有效输出，不因尺寸拒绝删除中心。方向与中心标志、框/score/数量/缺失保持。

| 组 | 中心命中/输出 | 输出覆盖 | 全帧中心正确覆盖 |
|---|---|---:|---:|
| all | 874/886=98.6456% | 886/887=99.8873% | 874/887=98.5344% |
| domain:real | 362/374=96.7914% | 374/375=99.7333% | 362/375=96.5333% |
| domain:sim | 512/512=100.0000% | 512/512=100.0000% | 512/512=100.0000% |
| sequence:real_seq07 | 213/225=94.6667% | 225/226=99.5575% | 213/226=94.2478% |
| sequence:real_seq14 | 149/149=100.0000% | 149/149=100.0000% | 149/149=100.0000% |
| sequence:sim_seq10 | 512/512=100.0000% | 512/512=100.0000% | 512/512=100.0000% |

**回传与审查。** 修正审查器后，服务器及本地逐帧PCA/标准化/网络/四状态/同数/同CR/tie/门限与来源闭包均通过：3445行、210组状态/匹配、120组AUROC，风险最大差服务器0、本地9.77e-15。追加诊断以独立Torch float64及标量中心/B-M检查30组复核通过；不拟合或改门限。旧审查进程因重复解压而结束，原shell退出130/旧日志保留；只重跑审查，模型训练与VAL评价未重跑，不能宣称原shell无错误退出。

统一结果目录 `work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result`，完整权重/PCA/raw特征/trace/报告保留。服务器 `independent_review.json`、本地 `mac_review.json`分别保留；两端共享 `source_comparison_analysis.json`及`analysis_review.json`。report SHA `9231b9344c3aa7665479f2ae2e0fcbd4df3a3678815c139fbae6d42e966c4203`，models SHA `338b7d514135a1e66dc371829ac5b2c8c060fd7ae0347c35174266d2de5b2851`，诊断SHA `270bc7eaa93de26d6bc9d13d32504864fedd85369080a4cf04ec94c558db2c8a`。

唯一回传包位于work_dirs根目录：`port_reliability_feature_source_v1_20261009_source_v1.tar.gz`，51720036字节、36成员（34文件），SHA `f61d76a2d451ea9a8de7da05ab4d678d84b6bde70d92f83af0405a05857ec9fe`。加入修正审查后重打包，原30文件全部逐成员字节核对保留；原归档SHA/退出状态记在 `archive_update.json`。本地SHA、成员路径/类型/体积及原产物回执核验，未覆盖已有目录。追加诊断/本地复核小文件随后同步在同一目录，没有新增本地结果副本。服务器已有工作区/暂存指纹保持b48cc0…/d54105…。分析记录同步后仅清理本次两端压缩包，完整结果保留，清理回执另记。

**决定。** 保留正式M/simple/policy，此candidate有总体高正确保留点与Real局部增量，但未形成超过simple的稳定逐域/逐视频优势，封存。它检验了规定的特征来源适配，不是完整ReDC复现，也不能据此否定原论文。后续不直接进入TEST或扩容/换激活再搜；可以先整理正式可靠性结果及这一有限对照的适用边界。如继续优化，需要能解释Sim退化和交付M质量的新证据/新合同，不能仅凭训练拟合或总体FA下降启动下一版。失败候选只写运行记录，不写为论文正式性能提升。

**分析同步与清理完成。** 分析提交 `dec7db3` 已在两端main同步。两端report/models/final_heads/PCA及诊断/复核SHA一致；在完整产物和原B24/σ1.5ep03/policy核验保留后，仅删除本次两端 `port_reliability_feature_source_v1_20261009_source_v1.tar.gz`。完整结果未删、未新增本地回传副本；两端同一RUN_ID内保留 `archive_cleanup_local.json`、`archive_cleanup_server.json`。服务器已有工作区/暂存指纹仍为b48cc0…/d54105…。新版本仅为审查器工程修正与记录，原训练提交fa1ab31/权重/门限/report身份保持，不补跑训练或TEST。

## 38. 第37节原生特征候选的冻结TEST补评（2026-10-09）

### 38.1 授权、实现及固定合同

用户在明确第37节总体高正确保留点改善、Sim局部退化之后，明确授权补齐TEST，沿用本地修改/检查、Git同步、指定服务器目录运行、回传独立审查、分析记录同步与本次压缩包清理流程。本项是另立的冻结诊断补评，不撤回第37节VAL_FAILED_STOP，不用TEST调参、选权、重选特征或自动采用；已暴露TEST不是未接触独立证据。

原生native与同容量midpoint控制均保留原epoch100/每臂1000updates，原TRAIN-only PCA256、标准化和四种原VAL单一全局整tie门限固定。原正式M=B24+σ1.5/epoch03、原policy及三标志身份保持；仅实验尺寸risk/flag可变化。不训练、不修框、不改变score、数量、中心或方向，不恢复深度/几何支路。

新增独立入口`eval_port_reliability_feature_source_test_v1.py`及CPU独立审查、固定protocol/sources、单测、`tools/run_port_reliability_feature_source_test_v1.sh`。旧TRAIN/VAL入口及其拒绝TEST角色的保护未改。新native特征通过原实际topk只读追踪完整TEST1440帧取得，核对真实retina_reg卷积、原B框/score与状态digest；midpoint特征复用第36节已审查且SHA固定的TEST stem288，与当前交付M描述量配对，使用第37节冻结PCA与网络。旧ROI、权重和TRAIN/VAL来源不复制到结果目录。

完整TEST按总体/域/视频报告FA、FR、ED、CR、MISSING、接受后错误比例、正确保留及全帧正确覆盖，同接受数score/simple/旧特征控制及同CR整tie参照、错误AUROC、连续FR与ED；各组离线匹配不能汇总成总体参照，也不生成线上TEST门限。中心只对有输出统计命中，另报输出覆盖与全帧正确覆盖。缺失帧无特征或尺寸判断，GT仅用于离线标签、身份核验和评价。

本地8项新增合同/数值检查、Python与shell语法通过；原特征来源15项单测在服务器一并运行，包含本地缺Torch未能执行的实际Torch检查。结果集中`work_dirs/port_reliability_feature_source_test_v1/RUN_ID/`，一个回传包在work_dirs根目录；结果分析与记录同步后仅删除本次两端包。初始实现尚无TEST数值，后续实测原位追加于本节。

### 38.2 完整TEST实测、独立回传核验与分析

**执行事实。** 源码提交`3ca5ae3`已推送并在指定服务器main快进拉取，mmrotljj/Python3.8.20/Torch1.13.1+cu117/NumPy1.24.4、物理GPU3（GTX1080）执行；GPU2有其他任务，未干预。23项服务器检查（新增8项+原来源15项）通过。本次Git拉取使用临时仅回环17890→本机7890的SSH转发，连接结束即撤销，未改持久配置。完整1440 TEST帧、1431输出、1089正确尺寸/342错误尺寸/9缺失。原B六项最大重放差全部0，冻结state保持；本次1440个原生特征forward只提取和核验，不生成替代交付框。无训练、选权或门限更新。

**沿用原VAL单一全局门限的实际使用点，不是公平同数比较：**

| 方法 | 接受数 | FA | FR | ED | CR | 正确保留 | 接受后错误比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| native候选 | 1369 | 313 | 33 | 29 | 1056 | 96.9697% | 22.8634% |
| 同容量midpoint控制 | 1335 | 274 | 28 | 68 | 1061 | 97.4288% | 20.5243% |
| simple评分／原VAL95门限 | 1360 | 287 | 16 | 55 | 1073 | 98.5308% | 21.1029% |
| score／原VAL95门限 | 1402 | 317 | 4 | 25 | 1085 | 99.6327% | 22.6106% |
| 正式原simple policy | 1330 | 275 | 34 | 67 | 1055 | 96.8779% | 20.6767% |

native相对正式policy多接受39帧，其中只多保留1个正确、却多接受38个错误；不能因FR少1宣称全面改善。native接受覆盖1369/1440=95.0694%，全帧正确尺寸覆盖1056/1440=73.3333%；正确保留的分母是1089正确输出，不是1440全帧。

**公平排序比较：**

| 全TEST参照 | native | midpoint | simple | score |
|---|---:|---:|---:|---:|
| 同1369接受：FA | 313 | 291 | 292 | 307 |
| 同1369接受：FR | 33 | 11 | 12 | 27 |
| 同CR1056：FA | 313 | 273 | 275 | 306 |
| 全TEST错误AUROC | 0.585945 | 0.611431 | 0.766600 | 0.591736 |

同数candidate比simple多21FA及21FR，比score多6FA及6FR，比同容量midpoint多22FA及22FR。同CR1056时比simple多38FA，四方法均可精确整tie匹配，不是边界tie选择产生的差异。所有同数/同CR参照只作离线评价，不写回TEST门限或线上policy。

**分域/逐视频，固定候选点与组内参照：**

| 组 | native FA/FR/ED/CR/缺失 | native接受数 | 同数simple FA/FR | 同数score FA/FR | 同CR native/simple FA |
|---|---|---:|---|---|---|
| Real | 291/12/29/527/9 | 818 | 284/5 | 289/10 | 291/267 |
| Sim／seq09 | 22/21/0/529/0 | 551 | 22/21 | 20/19 | 22/22 |
| Real seq03 | 83/4/19/87/7 | 170 | 79/0 | 83/4 | 83/47 |
| Real seq04 | 208/8/10/440/2 | 648 | 208/8 | 205/5 | 208/208 |

native Real正确保留527/539=97.7737%、Sim529/550=96.1818%、seq03 87/91=95.6044%、seq04 440/448=98.2143%，各组实际>=95%不能代替错误辨识验收。Real接受后错误291/818=35.5746%；seq03 83/170=48.8235%。Sim把22个错误全部接受，却拒了21个正确框；正式simple在Sim原点全接受，FA也是22而FR0。各组离线参照分别匹配，不能把组内FA直接相加为总体参照。

同CR最长连续FR：全TEST native3、midpoint6、simple3、score2；Sim native3、simple/score2。没有把连续拒错算作误拒，当前新增误拒确实属于正确尺寸框。

**新增发现与适用边界。** VAL总体同数收益（少5FA及5FR）未迁移至TEST（多21FA及21FR）；TRAIN错误AUROC1.0、VAL0.716610、TEST0.585945，当前冻结来源映射未形成稳定泛化。原生特征不是完全无信息：seq04 AUROC0.640359高于simple0.543445，但同CR440仍与simple相同FA208；seq03 AUROC仅0.527149，simple0.804999，同CR87多36FA。Sim native AUROC0.590826高于simple0.535950，也未在冻结使用点检出错误。这说明局部曲线信息、跨视频整体排序、实际使用点要分开；不能把TEST部分AUROC改善改写成候选可用，也不能从TEST另选控制A。是否由TRAIN支持、PCA/特征、质量映射或训练方式导致，尚未隔离唯一根因，不据本次结果调激活、维数、门限或追加预算。

**中心及三标志保护。** 原框、score、数量/缺失、center/angle标志逐帧保持；尺寸拒绝不删除中心。全TEST中心命中1430/1431=99.9301%，输出覆盖1431/1440=99.3750%，全帧中心正确覆盖1430/1440=99.3056%；逐域/视频三口径保存在report/analysis。9缺失不进入FA/FR/ED/CR，不创建尺寸特征或判别。

**回传与核验。** 唯一归档`work_dirs/port_reliability_feature_source_test_v1_20261009_source_frozen_test_v1.tar.gz`为19072039字节、30成员，SHA`0d2629cff9d63cf07660baf8910a4b1396c682a18b3c761b35bcfaef28de7da4`；回传SHA、路径/类型及原产物回执全部通过，解压到唯一同名实验/RUN_ID。两端独立重放1440行、1431输出、108组状态/同数/同CR与48个AUROC；risk最大差服务器0、本地1.3822e-14。服务器`independent_review.json`与本地`mac_review.json`分别保留，原产物不覆盖；本地初次CLI写已有审查文件触发no-overwrite保护，审查计算无失败，随后直接调用audit写mac_review完成。report SHA`7ef176a6a6d926fd11d4ca422039796381a75a88a08eaf08d1b3a2fbffa0f43b`，第37节模型/PCA/门限字节保持；完整权重、原缓存与本次特征/报告保留。

**决定。** 维持正式M/simple/policy，第37节候选仍不采用、VAL失败身份保持，不以TEST反选A或重调B。收束本次特征来源候选，不自动追加训练；本项失败诊断只记运行记录，不写为论文正式性能提升。已经完成的正式三分量成果可继续用于写作，但本候选不能称为稳定升级。本次冻结TEST补评退出码0，服务器原工作区/暂存指纹保持b48cc0…/d54105…。分析同步与本次两端压缩包清理回执随后追加。

**分析同步与清理已完成。** 分析提交`b85a754`在两端main同步；服务器已核验全部原B24/σ1.5ep03/模型/PCA/policy/门限及原结果回执，mac_review SHA`01ae30ed4fe293a6f0d2925dee07b12dd02f291568c1600719858dd6c8042e43`、analysis SHA`04353d423c693581e8ebd27f4e5d19708fcd847677780bc5adb438db83cb700d`两端一致。仅删除上述本次两端唯一tar.gz，完整TEST结果、原模型及缓存保持；同一RUN_ID保存并互传`archive_cleanup_server.json`/`archive_cleanup_local.json`，原工作区/暂存指纹保持。本地已有无关删除/未跟踪文件未纳入本次提交，未修改正式成果或其他窗口。

## 39. 同图尺寸框条件监督有限对照（2026-10-09）

### 39.1 授权、固定设计及运行入口

用户认可先前设计，授权本地实现、Git同步及在服务器`/media/omnisky/personal_files/ljj/symEOOD`使用后几张空闲卡运行、回传分析。原正式M=B24+σ1.5/epoch03和simple policy保持；主线是尺寸可靠性判断，不修框、不改score/输出/中心/角度，不恢复几何或深度训练。ReDC提供细分定位质量判断动机；本项同图尺寸对照监督属于项目适配，不是原ReDC CR/DDE/CAR复现。RARE只参考相对质量排序的评价思路；首轮不叠加其成对损失或候选检索。

**两臂。** `original`仅真实完整TRAIN监督；`contrast`为相同真实损失加0.25辅助BCE。两臂都在被判断框自身提取冻结P3 aligned9x9 ROI，1.5上下文、channel-major3x3均值2304维；只用真实TRAIN M拟合共享PCA256及259维标准化，头均为258→16→8→1 ReLU+beta共4290参数，risk logit=-原score logit+残差。与第37节固定B winning cell/第36节midpoint stem的输入不同，旧候选仅历史参照；本项唯一两臂差别是辅助监督，不把与旧候选差异归因于这一因素。

**对照框。** 原始TRAIN M之外，分别长边/短边/两边共同乘0.85、0.95、1.05、1.15，最多12个。生成API无GT，保留原raw w/h关联、中心/角度/score；导致长短轴次序反转或非有限/非正尺寸则跳过。GT随后离线计算每框双边相对误差是否均≤10%，不依据GT选择倍率或修成正确框；合成错误不等于新增真实场景或独立短边标注。辅助项不加类别权重，先按每图有效对照数平均，再按真实图数量平均；原项沿用完整TRAIN N/(2Nclass)权重。VAL不生成对照，最终只评价原M框。

**预算及预检。** seed1701，同初始化、同真实图排列、batch256，各100epochs/1000updates，固定末轮权重。两步smoke丢弃；记录原项、0.25加权辅助项的梯度范数、比值、合梯度及裁剪前后变化，训练每epoch首batch同样记录。首步零末层会使早期层梯度为零，第二步必须有实际隐藏层和辅助梯度；不恢复旧逐元素严格梯度等式比较。辅助计算增加开销，同更新数不声称同FLOPs。不给网络GT、视频或域身份，不改变激活/PCA维数/权重/预算再搜索。

**数据与继续条件。** 原完整TRAIN2558输出/71尺寸错误；完整VAL887帧/886输出。没有新增fit/probe等角色。VAL单一全局整tie门限保护各域及逐视频正确保留≥95%；`contrast`在总体同接受数及同CR均严格优于新`original`、simple与score，分组FA不增加、同CR连续正确FR不增加；保留FA/FR/ED/CR/MISSING、覆盖、AUROC和中心三口径。失败停止；成功才进入冻结已暴露TEST报告，不从TEST选方法或调参，不自动采用控制臂。

**身份与审查。** 使用已有原始M逐帧记录及policy；在线特征采样无GT。服务器原B与M重放核验、状态digest、输入SHA前后保持，原M值不被重放值替换。CPU独立审查重新生成框/标签并复算PCA、网络、工作点、同数/同CR和2000条更新。跨NumPy/libm只容许policy风险≤1e-12的末位重放差，保存的风险与实际接受标志逐帧不变；检测框/分数及标志身份不放宽。回传只复制本项新池化特征、模型JSON、记录与报告，检测器/旧ROI缓存/.pth不进入包；小型原cache manifest快照保留供无本地权重审查。

入口：`tools/run_port_reliability_box_contrast_v1.sh`，优先物理GPU3再GPU2，显存<200MiB且利用率<5%才启动；原工作区/暂存指纹保存。所有日志/结果集中`work_dirs/port_reliability_box_contrast_v1/RUN_ID/`，唯一归档在work_dirs根目录。mmrotljj环境运行：

```bash
cd /media/omnisky/personal_files/ljj/symEOOD
git pull --ff-only
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
unset CUDA_VISIBLE_DEVICES
bash tools/run_port_reliability_box_contrast_v1.sh
```

实现阶段尚无性能结论；实际执行、独立审查和决定在本节后续追加。原未采用候选不写为论文正式收益。

**启动前工程修复。** 首次`20261009_box_contrast_v1`在服务器单测阶段停止，未提取TRAIN、未更新模型。填充隔离测试的`az`由`expand`共享原score张量，向填充位置写NaN会污染真实score；修正夹具为独立`clone`，损失实现、设计、数据和训练预算未改。首轮失败日志保留，后续另用新RUN_ID运行，禁止覆盖或把单测失败算作性能结果。

**报告接口修复。** `20261009_box_contrast_v1_fix1`服务器39项检查通过；两臂各1000updates完成，完整VAL887帧和原始M评分/门限、末轮权重已保存。生成隐藏层诊断时误调用旧291维标准化，触发`Invalid fixed normalization`。修复为本项259维诊断，新增对应回归检查；使用单独`finalize_port_reliability_box_contrast_v1.py`只恢复已完成结果的报告，旧权重/评分/门限/日志SHA前后保持，不重训、不重复推理、不调门限。训练来源固定`6a03c336f98986bbe899b2ceaff042392fb2b369`，报告修复来源另记。原collect的B/M逐帧容差核验已通过，但未落盘的实际最大差和总耗时不能恢复，报告填null并披露，不编造为零。原失败文件和退出码保留，最终报告及审查完成状态单独记录。

### 39.2 完整TRAIN/VAL结果、回传复算及决定

**执行事实。** 服务器原工作目录、mmrotljj/Python3.8.20/Torch1.13.1+cu117/NumPy1.24.4、物理GPU3运行。原工作区/暂存指纹保持`b48cc0…`/`d54105…`，未干预GPU0/1/2任务。两臂各100epochs/1000updates，真实TRAIN2558框/71错误、有效辅助30370框/15838错误，2555图有正确/错误对照组合；全部有效对照的池化特征确实变化。39项初检通过，报告修复25项通过（含新增维数回归，共40项不同检查）。原失败退出码1保留，恢复报告和独立审查退出码0；性能状态为`VAL_FAILED_STOP`，工程状态与性能分开。

**完整VAL887帧/886输出、663正确尺寸/223错误尺寸/1缺失：**

| 公平总体参照 | 对照候选contrast | 同输入真实框控制original | simple评分 | score |
|---|---:|---:|---:|---:|
| 同871接受：FA | 211 | 214 | 208 | 209 |
| 同871接受：FR | 3 | 6 | 0 | 1 |
| 同CR660：FA | 211 | 216 | 202 | 201 |
| 错误AUROC | 0.586152 | 0.669710 | 0.785626 | 0.684394 |

候选固定VAL点FA211/FR3/ED12/CR660，接受后错误211/871=24.2250%。原正式policy原点FA190/FR10/ED33/CR653，两者接受数不同，不作为公平排序证明。候选同数比A少3FA、同CR少5FA，但分别比simple多3及9FA，比score多2及10FA，不能以优于新控制代替优于当前正式评分。

| 分组同正确保留 | CR | 候选FA | 控制A FA | simple FA | score FA |
|---|---:|---:|---:|---:|---:|
| Real | 196 | 167 | 167 | 156 | 157 |
| Sim/seq10 | 464 | 44 | 48 | 45 | 44 |
| Real seq07 | 60 | 154 | 151 | 130 | 144 |
| Real seq14 | 136 | 13 | 12 | 13 | 13 |

组内离线参照不可相加当总体参照。各组正确保留≥95%通过，但seq07同CR比simple多24FA；Sim局部只比simple少1，和score相同。候选同CR最长连续正确FR2帧，A/simple/score均1帧；27项关联门槛失败，不用失败项数代替实际指标。缺失1帧只记MISSING，无框无risk、不计FA/FR/ED/CR。

**新增发现（事实）。** 真实TRAIN两臂错误AUROC均1.0；合成同图配对AUROC由A0.554460提升至0.918894。辅助目标实际生效：100个epoch首batch的加权辅助/原项梯度范数比0.081158～0.535417，中位0.270339；两臂所有更新都未触发clip5，最大合梯度分别2.28205/2.34134。0.25不是被裁剪抹除，亦不证明它最优。VAL隐藏层全零输出由655降到133、零隐藏错误由172降到46，实际VAL错误排序仍退化，不能把问题只归因于隐藏层未激活。由于TRAIN实际点使用各自VAL门限，不能拿候选TRAIN实际FA68与A0单独宣称TRAIN排序失败；两者TRAIN AUROC均1.0。

**推断及边界。** 当前合成尺寸质量响应没有形成真实VAL错误辨识收益；增加有效梯度、样本变化和激活数量没有自动带来泛化。尚不能隔离图像特征、标注/真实错误分布、质量映射或辅助目标的唯一根因；配对成功也不能证明学到了独立的视觉边界证据。不能宣称ReDC原方法一般不适用，更不能从本次失败反向调整倍率、.25、激活、容量、门限或追加预算。

**保护。** 原M框、score、输出/漏检、中心及方向标志逐帧保持，尺寸拒绝不删除中心。VAL中心命中874/886=98.6456%，输出覆盖886/887=99.8873%，全帧中心正确覆盖874/887=98.5344%。没有深度或几何性能变化声明，无TEST读取。

**回传及独立审查。** 唯一最终包`work_dirs/port_reliability_box_contrast_v1_20261009_box_contrast_v1_fix1_final.tar.gz`365580742字节、41成员，SHA`c9a92e67d2dbc0b0ec0a62cb8efb31aa3c539bf9d653655a9fa5dfcd71a4ea40`，路径/类型/SHA核验通过，解到唯一同名RUN_ID；仅含本项新池化/投影特征等，不含.pth及旧ROI缓存。服务器及本机完整复算3445行、30370对照、2000更新、210组状态/同数/同CR、120个AUROC通过；risk最大差服务器5.50e-15、本机4.55e-15。审查器补充本地缺失返回包.pth的明确排除核验（服务器已验字节与模型JSON），并固定类别权重float64，避免NumPy1/2标量提升差异；不改已保存报告。report SHA`ee86076634f177a29ff2e6a1019950c72134930a3b2e24aef3003e7a1c145b95`、models SHA`5f9b7cbf400bf223a12107f7726f4fdc3441d795e8972c5bea42a8553d5d09ca`。报告训练来源和报告修复来源分开保留，不重写原失败事实。

详细分析集中`work_dirs/port_reliability_box_contrast_v1/20261009_box_contrast_v1_fix1/result/analysis.md`及`analysis.json`。**决定：本候选不采用、VAL_FAILED_STOP，不进入TEST，不反选A；正式M/simple/policy保持。** 失败只记运行记录，未改论文正式结果。后续先整理已有正式可靠性成果及负结果边界，不自动启动新候选。Git分析同步、当前审查器双端复核及本次压缩包清理回执后续追加。

**分析同步、当前复核和清理已完成。** 两端main在`744a378`同步分析及当前审查器，双端当前复算3445行/30370对照/2000更新、210组状态/120个AUROC及全部27项门槛失败完全一致。`mac_review_current.json` SHA`0420cca3015e70fd275ab12d02edbe7c1ff6da910c8777247038d26066d3d16a`、`server_review_current.json` SHA`fab1c5295f9468e620499f0d1b418df4499e633630dd72e1fb6113ce02100af6`；analysis JSON SHA`03864c5ba103fa2fc7e1347bfbc541ecd98fa3512eef082958f26188d00762d1`。服务器仅删本次初始单测失败包、报告失败包和最终回传包三个精确SHA文件；本机仅删唯一最终回传包。所有结果/原失败日志、服务器末轮.pth、PCA/模型JSON、原M/simple/policy均保留，server/local清理回执已互传至同一result目录。原工作区和暂存指纹仍`b48cc0…`/`d54105…`；本机原有无关删除/未跟踪文件未纳入任何提交。后续仅同步此收尾记录，不需再运行训练或评价。


## 40. 冻结图像评分对simple的互补性核对（2026-10-09）

### 40.1 范围与预定合同

用户授权沿用本地代码/检查→Git同步→服务器原工作目录运行→回传独立复算→记录→清理本次压缩包的流程。此项只核对历史评分，不训练/重新推理/提取特征，不读取TEST，不拟合融合器或搜索门限；正式M、simple及三个分量接口保持。各失败候选原状态和身份保持，不因本次离线交集反向采用。

复用第35、37、39节六个固定评分（readout_temperature/readout_residual、source_midpoint/source_native、box_original/box_contrast），原完整TRAIN2558帧/71错误及VAL887帧/886输出/663正确尺寸/223错误尺寸/1缺失。每个历史分支都保留，既有report、models、completion、逐帧评分、原始TRAIN/VAL表和正式policy以SHA绑定。主要基准为各历史报告共同保存的simple各组正确保留≥95%单一VAL门限，正式policy原门限另列；所有历史门限原样应用，不重新标定。

逐域/视频报告FA/FR/ED/CR/MISSING、接受与正确覆盖、连续正确FR，并分解：simple接受错误而候选拒绝的补充检出、simple保留正确而候选拒绝的新增FR、恢复CR及新增FA。报告simple已接受集合内的条件错误AUROC，与同集合simple/score比较。既有两规则交集（AND）仅作冻结的额外拒绝诊断，完整报告同接受数、同CR simple/score及边界并列风险上下限；GT可挑出的单支/六支联合互补上限单列，不是在线规则或性能收益，各组结果不可相加作为总体。

**必要继续证据（不是模型采用）：** 在simple正确保留95%基准上，固定AND必须同时满足VAL总体、各域、各视频正确保留≥95%；同接受数及精确同CR FA不劣于simple/score且总体严格改善；连续正确FR不劣于同CR对照。完整TRAIN仅为已拟合样本诊断，不算独立验证。VAL已多次开发暴露，本项不声称独立泛化或统计风险保证；即使存在通过项也只支持后续设计，不自动训练/采用。若无通过项，正式M/simple保持，不自动训练融合头或进入TEST；局部互补事实和保护代价仍须保留，不能宣布所有图像信息或融合均无效。

实现入口`crane_project/tools/audit_port_reliability_complementarity_v1.py`，独立标量/配对AUROC复算`review_port_reliability_complementarity_v1.py`，合同/来源清单同前缀JSON。统一运行`bash tools/run_port_reliability_complementarity_v1.sh RUN_ID`，纯CPU、不占GPU；原服务器路径和mmrotljj环境。全部新日志/结果位于`work_dirs/port_reliability_complementarity_v1/RUN_ID`，只打包本次运行至work_dirs根目录，不带旧缓存或权重。

本地15项针对性测试、脚本语法、输入SHA/完整3445帧身份及统计逻辑预检通过；此处尚无服务器执行或独立审查完成声明。结果及最终决定在40.2追加。未改论文主要结果文件。

### 40.2 服务器结果、互补性发现与决定

**本次互补性核对完成；存在局部互补信息，但没有跨域稳定的固定叠加收益，不采用新policy、不自动训练融合头。**

完整TRAIN2558帧及VAL887帧逐帧身份、既有评分和门限保持，六个分支全部核对，0训练更新/0检测推理/0新特征前向/0门限拟合/无TEST。两端15项测试及独立3445行、180组、1980项状态/匹配汇总、540个条件AUROC复算一致。

主要基准是历史已冻结的simple各组正确保留95%点（FA200/FR7/ED23/CR656，接受856）；不是正式policy原点FA190/FR10/ED33/CR653，不能把两者混淆。

| 固定原生特征与simple交集，分别同正确保留 | CR | 交集FA | simple FA | score FA |
|---|---:|---:|---:|---:|
| 总体 |647|178|188|187|
| Real |190|136|152|144|
| Sim |457|42|35|42|
| seq07 |60|123|130|144|
| seq14 |130|13|13|13|

每行是该组独立的离线同CR比较，不能把组内对照相加作为总体。同接受825个框时，总体交集FA178，simple184、score185；Sim同接受499个框时交集42、simple40、score42。并列风险边界已报告，无GT挑选并列成员。

**事实。** 原生特征对simple95的22个补充检出分布为Real18（全部seq07）、Sim4；9个新增FR为Real2、Sim7。交集FA200→178、FR7→16，正确保留647/663=97.59%；逐域/视频正确保留均≥95%，连续正确FR最长1帧，保护通过。其必要证据门槛只在Sim/seq10同数及同CR与simple比较失败（重复组，不是四个独立问题）。因此不是仅靠覆盖变化：总体/Real/seq07同CR存在实际局部收益，但Sim同CR的FA42>35抵消了跨域稳定性。

**排序边界。** 在simple已接受集合内，原生分支条件错误AUROC为总体0.6899、Real0.6961、Sim0.6756、seq07 0.4897、seq14 0.4929。对应simple为0.7693/0.5011/0.6801/0.4782/0.4343。Real汇总改善不等同逐视频稳定辨识：seq07自身条件排序仍接近随机；固定尾部额外检出18个不支持宣布整体排序可靠。不能由此宣布图像特征、PCA、标签或ReDC适配的唯一根因。

**最近对照分支没有补足该缺口。** box_contrast对simple95额外检出7个错误但新增FR3；固定交集FA193/CR653，而同CR simple为190，并且Real/seq07正确保留和连续FR保护失败。六个固定交集均未通过必要门槛，温度分支虽有总体局部增量，仍在Real/seq07公平比较失败。不能把局部排序或较少FA单独当采用依据。

**Oracle不是收益。** 六支联合用GT挑选可覆盖40个simple接受错误，以及恢复7个误拒正确框；这是选择有利改动的离线上限，不是一个不读GT的在线融合器，也不是最佳权重/模型选择证据。TRAIN模型都已用该TRAIN拟合，原生分支在TRAIN补充检出14且新增FR0也不能作为泛化保证。

**建议与边界。** 本次核对回答了“是否完全没有补充信息”：不是，原生特征在部分Real样本有互补；也回答了“可否直接叠加”：当前固定规则不满足跨域保护，不采用。保留正式M/simple。下一步不追加合成扰动、不反选域专用门限、不按seq07/Real手动切换、不自动运行TEST或训练残差/融合头。若以后另获授权继续，只能先提出在TRAIN中学习且保护simple原有正确观测的明确机制，并同时验收Sim；本次不足以保证该机制有效。正式可靠性成果可继续整理，负结果与局部线索保留原身份。

服务器原路径、mmrotljj/Python3.8.20、纯CPU执行。GitHub专用代理失效，命令级绕过后直连443亦超时；使用仅含两个提交的Git bundle经SSH快进拉取，没有改服务器代理配置。自动审批一次超时未执行，拆分同步/运行后完成。工作区及暂存改动SHA保持。结果包SHA 4d399c0468fadd612bb47efb81e0f5c113b2f53140d4b246cee0a59fc16f8b8c，4882185字节、17成员，仅本次报告/合并评分/日志，无权重、旧缓存。完整报告SHA ec636fa51dcc97870f570642c383a70935cc22b88c6f9998778994ec1e41002f，两端复算一致。完整结果保留；压缩包清理另有回执。

### 40.3 收尾与来源身份

本项代码执行提交`c1b4ee9eedad7e7bcfd1fa270a942e559ab4af32`。服务器及本机各15项针对性测试通过；两份独立审查JSON逐值一致，均核对3445行/180组/1980项汇总/540个条件AUROC，报告SHA一致。服务器工作区/暂存指纹分别`b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1`/`d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e`，同步及执行前后保持；本机没有无关改动纳入本项提交。现有用户提交`51c25ae`作为既有历史一并快进同步，未将其中整理动作冒充本项修改。

`interpretation.md`/来源JSON、两端审查、回传安全核验及两端清理回执全部集中在唯一RUN_ID/result，两端已互传。结果原包4882185字节、17成员、安全路径/文件类型/SHA核验通过；只有合并评分/统计/日志，无权重或旧ROI缓存。本次结果压缩包及首个临时Git同步bundle两端均核验精确SHA后删除，完整result/日志保留，原TRAIN/VAL评分、policy及历史模型输入SHA再次核对保持。`mac_cleanup.json`、`server_cleanup.json`记录精确文件清单；不清理其他包或旧证据。

最终运行状态`NO_STABLE_FROZEN_OVERLAY_KEEP_SIMPLE`、工程审查通过、六个必要证据门槛均失败（重复/关联条件不是独立失败个数）。正式M/simple/三标志保持，未拟合/采用融合policy，未进入TEST，未改论文主要结果文件。记录同步是Git快进传输，不重复运行核对；GitHub网络故障下使用Git bundle，不改服务器代理。临时记录同步bundle在完成身份验证后清理，回执另保存在同一结果目录。负结果不抹除原生分支Real局部互补信号，也不证明其他融合机制必然无效。

## 41. 受限图像残差的锚点对照（2026-10-09）

### 41.1 授权范围与运行前冻结合同

用户授权沿用本地修改/验证→Git提交同步→服务器原工作目录/mmrotljj→后两张空闲GPU→回传独立复算→运行记录→仅清理本次压缩包的流程。目标仍为冻结M上的尺寸可靠性；不修框、不改score/输出数量/中心与方向，不重启深度或旧特征路线。本实验不是第40节AND规则采用，亦不读取旧训练神经头评分作融合输入。

**依据与待验证机制。** 第40节原生特征对simple有Real局部互补，但交集在Sim公平比较退化；Real pooled条件AUROC改善不代表视频内有效。本项仅检验：将图像残差限制在simple附近，能否保留原评分基础并获得真实错误辨识增量。尚未认定该机制有效或已查明唯一根因。

两臂A=`score_anchor`、B=`simple_anchor`，分别以`1-delivered_M_score`及保存的正式simple尺寸risk为锚点；相同原生B回归输入patch/PCA256、259维TRAIN标准化参数，排除score descriptor后的258→16→8→1 ReLU+beta，共4290参数。只复用第37节已核验原生投影缓存、TRAIN-only PCA及标准化，不重新检测/提特征、拟合PCA，不复用历史神经头权重。特征/历史来源/原始2558+887帧表/policy全部SHA绑定。

共同读出`sigmoid(logit(r0)+0.5*tanh(g(x)))`，0.5预先固定，两组均为受限残差；只有锚点不同。为精确保持保存的FP64 simple基准，两组head/输入/锚点/损失统一FP64，与历史FP32自由残差另保留身份，不声称逐位复现历史训练。中性风险用可微等价式`r0*(1+expm1(delta))/(1+r0*expm1(delta))`；不使用零值分支截断梯度。实际锚点严格(0,1)，不临时clip。Torch种子1701标准初始化后转FP64、deepcopy；输出weight/bias和beta为零，discarded两步smoke后重建同初始化，验证全TRAIN/VAL零risk与正式simple原标志精确保持。

完整真实TRAIN2558输出/71错误，100epoch、batch256、每臂1000更新/共2000；同顺序种子1701。仅原class-balanced genuine size BCE，AdamW .001、权重wd1e-4、bias/beta无wd、clip5；不加合成框/辅助或pairwise目标/视频域权重/采样变化。记录每步loss、分参数梯度、clip前后及有界残差/饱和比例。只评价最终epoch100，无epoch选择；VAL特征在最终head冻结后加载。TRAIN是拟合诊断，VAL多次开发暴露，不能称独立泛化。

完整VAL887帧/886输出/663正确/223错误/1缺失，按各评分分别一次全局whole-tie门限，使总体/两域/每视频正确保留≥95%；原正式95%全帧覆盖点单列。B必须在总体同接受数、精确同CR均严格优于A/simple/score；逐域/视频同数保守并列下限及同CR FA不增、最长连续正确FR不增；至少一个Real视频同CR严格优于simple。分域逐视频FA/FR/ED/CR/MISSING、接受/正确覆盖、正确保留、错误比例、AUROC及连续FR均报告。组内参照不相加；拒绝连续错误不计连续正确FR。无框不作判断，中心另报输出帧命中/输出覆盖/全帧正确覆盖。

失败即`VAL_FAILED_STOP`，保持正式M/simple，不反选A、不追加预算/改.5或门限/扩容量/改激活、不进入TEST；只有预定B全部通过才允许冻结TEST报告，TEST已多次暴露且不用于候选选择。在线接口不接GT/视频/域；GT仅监督/离线评价。未采用实验只记本运行记录，不写论文主要结果。

入口`crane_project/tools/run_port_reliability_simple_anchor_v1.py`及同前缀独立审查/合同/来源JSON；统一命令`bash tools/run_port_reliability_simple_anchor_v1.sh RUN_ID`。GPU3优先、其次2，均忙即停止。所有日志/结果集中`work_dirs/port_reliability_simple_anchor_v1/RUN_ID`，仅本次结果包置work_dirs根，不含.pth或旧ROI/投影缓存。

**已完成本地检查，尚未运行服务器。** 17项NumPy/接口/门槛检查通过；5项Torch梯度/等价读出/同初始化检查因本地无Torch明确跳过，服务器必须全通过。完整3445帧来源/标签角色、投影配对、TRAIN矩及policy复现、所有实际锚点有效及脚本语法预检通过。服务器工程及性能结果在41.2追加。

### 41.2 完整TRAIN/VAL运行、发现与决定

**两臂已运行、双端独立复算通过，预定simple锚点候选失败，不采用、不进入TEST。** 执行提交`ec990da5753e71cb37d826c9fa2d17d83ea136b5`，服务器原工作目录/mmrotljj/Python3.8.20/Torch1.13.1+cu117/NumPy1.24.4、物理GPU3。22项服务器检查、两步discarded smoke通过，两臂各100epoch/1000更新，共2000；同FP64、初始化、曝光和预算。0检测推理/0新特征前向/0新PCA/无新数据角色。工程状态与性能`VAL_FAILED_STOP`分开。

完整VAL887帧/886输出/663正确/223错误/1缺失。各评分一次全局正确保留95%工作点：A(score锚点)FA202/FR4/ED21/CR659，接受861；B(simple锚点)FA190/FR11/ED33/CR652，接受842；simple FA200/FR7/ED23/CR656，接受856；score FA201/FR3/ED22/CR660，接受861。原正式policy原点另为843接受、FA190/FR10/ED33/CR653，保持原门限身份。不同接受数下200→190不是排序改善证明。

| 公平总体参照 | B simple锚点 | A score锚点 | simple | score |
|---|---:|---:|---:|---:|
| 同842接受FA |190|193|190|193|
| 同CR652 FA |190|195|190|196|
| 错误AUROC |0.786032|0.695033|0.785626|0.684394|

| 分组独立同CR | CR | B FA | A FA | simple FA | score FA |
|---|---:|---:|---:|---:|---:|
| Real |191|144|151|154|144|
| Sim/seq10 |461|46|44|40|44|
| real_seq07 |61|131|147|131|144|
| real_seq14 |130|13|13|13|13|

各组正确保留≥95%及连续正确FR最长1帧通过，但总体未严格优于simple，Sim同数/同CR比A/simple/score更差，两个Real视频均无同CR增量。Real pooled比simple少10FA不能视作视频内改善；组内参照不可相加。15项关联门槛失败包含Sim与seq10重复，不是15个独立问题。seq07 AUROC .539193→.561434，seq14 .415158→.403281，Sim .693382→.681618，不能以微小总体AUROC变化声称收益。

**新增事实。** B TRAIN错误AUROC .956823（simple .801752），A .767507（score .556528），梯度连通/真实TRAIN学习生效。两臂0次clip5，最大梯度0.80953/0.36403。B TRAIN2521/2558=98.55%达到abs(delta)>.49；VAL740/886=83.52%达到负向delta<−.49，874个输出降risk、12个升risk；223个错误中216降risk、仅7升risk。正确/错误VAL delta中位−.498405/−.497307。TRAIN实际B FA0/FR995是应用VAL门限的拟合诊断，不是可靠性成功，也不能忽略排序改善直接称TRAIN拟合失败。

**推断及限制。** 受限残差在VAL主要近似统一降低risk，未稳定识别实际错误；simple排序基础大致保持，未取得跨域和视频内所需增量。BCE与固定有界读出的饱和是本合同限制线索，尚不是唯一根因，不能据此改.5、损失/激活/容量或追加预算。不能宣布图像信息一般无用或ReDC原方法不适用。下一步保持正式版本，收束本受限残差候选，不自动重开旧支路；论文可整理正式结果和真实边界。

**核验、保护及回传。** 3445帧来源/中性风险及标志精确保持；双端独立复算3445行、435项实际/匹配汇总、60个AUROC和所有门槛一致。NumPy重放最大差服务器2.22e-16/本机2.78e-16，保存重载一致；report SHA`279545ab058397707def9a44e80d754ff0c3807d697e77dfc6e98af36b4ddcdc`，服务器末轮.pth SHA`eb1db7231db99ac9491662f7164cec39e35d12cdc128cac65c862b11385724e7`。框/score/数量/中心和方向/缺失身份保持；VAL中心874/886命中、886/887输出覆盖、874/887全帧正确覆盖。无GT在线/TEST/深度或几何修改，无新policy采用。

所有证据集中`work_dirs/port_reliability_simple_anchor_v1/20261009_simple_anchor_v1/result`，analysis.md为冻结报告生成、interpretation.md/json为事后解释、两端review及转移核验回执分开。唯一包2320698字节/24成员、SHA`9c676034b575f35676a18a384e353b7122c6d49f9086d7f64893439b7a29cf95`，安全路径/类型/SHA验证通过，只有本次评分/模型JSON/日志，不含.pth、旧投影或ROI缓存。未改论文主要结果。服务器GitHub不可达沿用单提交Git bundle快进，不改代理；原工作区/暂存指纹保持`b48cc0…`/`d54105…`。完整结果与服务器权重保留，包/bundle清理回执及最终Git同步后追加。

### 41.3 回传解释及清理收尾

独立分析interpretation.md/json、mac_review及安全回传回执均已传回同一服务器result目录并核对SHA；两端完整结果保留，服务器唯一末轮.pth保留。本机与服务器仅删除本次结果包SHA`9c6760…`及代码同步bundle SHA`db1d03…`，两份精确清理回执已互传。原policy/投影/PCA/历史输入的SHA经双端审查保持，原工作区与暂存指纹保持。候选状态仍`VAL_FAILED_STOP`，未用TEST、没有采用新policy或反选控制A。

分析记录另作Git提交推送并快进同步，不重跑训练或评价；最后记录同步bundle在两端确认HEAD/文档SHA一致后删除，另存回执至本项result。所有本项文件集中在一个版本/RUN_ID，未新增本地历史结果副本或论文正式成果记录。

## 42. 同视频真实尺寸错误排序有限对照：设计与本地实现（2026-10-09）

### 42.1 范围、依据与冻结合同

**状态：已实现并完成本地静态/NumPy验证，未提交Git、未推送、未连接服务器、未进行Torch实际训练，尚无性能结论。** 用户本轮授权按已认可设计修改代码；本节是新版本合同，不改变第41节失败事实，不续训其权重。主线仍为固定M上的尺寸可靠性判断，其他两个使用标志保留。

**事实与假设分开。** 第40/41节显示Real pooled排序不能代替视频内辨识，受限残差在VAL主要降低risk。新版本只检验同视频真实正确/错误之间的排序监督是否提供增量；尚未认定BCE是唯一根因，不保证泛化、消除饱和或补足Sim错误支持。文献依据为[NeurIPS 2025选择性预测分析](https://proceedings.neurips.cc/paper_files/paper/2025/file/71460926102fade443ea7ec89ae8a73a-Paper-Conference.pdf)和[2025可微AURC预印本](https://arxiv.org/abs/2505.23463)。下述视频内softplus损失属于项目适配，非两文原方法完整复现，未实现SoftRank，不继承其结果或统计保证。

保持M=B24＋midpoint sigma1.5/epoch03及匹配正式simple/policy。仅新增候选尺寸risk/flag，不修框、改score或输出数量，不删除可用中心、不更改方向，不恢复几何/深度、跨尺度、合成尺寸框或新特征提取。无检测框不作判断，risk=None；MISSING不计FA/FR/ED/CR。

两臂共同复用第37节已核验B原生实际回归输入patch、TRAIN-only PCA256与259维标准化；排除score descriptor后的258维（PCA256＋M相对尺寸/比例）进入258→16→8→1 ReLU＋beta，共4290参数。只复用缓存/PCA/矩，不复用历史神经头权重或评分；锚点均为正式simple尺寸risk。共同logit为`logit(r_simple)+0.5*tanh(g(x))`，risk仍用精确中性可微expm1式。FP64，seed1701，Torch标准初始化后转FP64并deepcopy；末层及beta为零。固定0.5，不按新结果放宽。

真实标签保持：最终交付M的规范长短边与GT中任一相对误差>10%为错误。GT、标签、video索引仅离线监督/评价，在线函数不接GT/域/video。完整TRAIN2558输出/71错误/2487正确；完整VAL887帧/886输出/663正确/223错误/1缺失。保持完整TRAIN→完整VAL→冻结TEST，不新增fit/probe等角色。

### 42.2 配对、损失、预算与工程检查

所有配对均为同一TRAIN视频内真实错误集合B_v与真实正确集合G_v的完整笛卡尔积，不跨视频、不按分数挖掘、不采样、不生成辅助框。输入按冻结image身份排序，索引和计数写入pair_manifest.json。

| TRAIN视频 | 错误帧 | 正确帧 | 同视频配对 |
|---|---:|---:|---:|
| real_seq01 |1|338|338|
| real_seq05 |44|516|22704|
| real_seq06 |13|453|5889|
| real_seq12 |5|136|680|
| real_seq13 |7|297|2079|
| sim_seq08 |1|747|747|
| 总计 |71|2487|32437|

32,437配对仍只来自71个错误帧，不是新增独立场景/错误事件。Sim及real_seq01各只有1个错误，按视频均值可能过拟合这些单个错误，必须披露并保护VAL Sim。静态检查：simple中12,037个反序/并列配对，只有1个在理想±0.5残差的最大两样本差<1限制下无法翻转；这只是逐对理想可行性，不保证共享网络能同时翻转其余配对。

`L_rank = mean_v mean_(b,g in B_v×G_v) softplus(s_g-s_b)`，s为最终风险logit。A=`bce_control`，仅原完整TRAIN类平衡真实BCE（N/(2N_class)）；B=`within_video_rank`，同BCE＋固定0.25 L_rank。各视频内平均，再6视频等权平均，不按配对数量平均。统一平移所有s不会改善rank项；仍不保证BCE不会驱动偏移、tanh不会饱和。

每臂100epoch、batch256的BCE、1000更新（总2000），相同seed1701真实帧顺序，每epoch最后BCE批254帧；AdamW lr .001、weights wd1e-4、bias/beta无衰减，clip5。每步两臂在全部2558缓存行做相同头部前向并计算全部32437配对；A仅记录rank，不参与正式更新，B反传0.25 rank。相同正式更新数和头部前向，不声称相同反向计算成本；没有新增检测/冻结图像特征前向。只选最终epoch100，无最佳epoch选择。两步独立smoke均丢弃，再重建同初始化/空优化器。

预检包括标签/配对身份、排序梯度方向、常量偏移不变性、完整头部有限差分、控制臂无rank梯度、候选BCE＋0.25rank实际更新、二步隐藏层梯度、中性精确还原、保存重载及Torch/NumPy重放。每epoch首批分别记录BCE与加权rank参数梯度范数、相对强度、余弦、分参数梯度及同一clip对两分项的共同缩放；每步记录总梯度裁剪前后、各视频rank、残差饱和、完整前向行数及rank是否参加更新。系数不据loss数值认定合适，也不因新梯度结果现场调参。

### 42.3 VAL验收、输出和停止条件

固定末态保存/重载完成后才在训练入口加载VAL特征进行评价。prepare可只读检查完整输入配对/来源/中性标志，不参与拟合。一次全局whole-tie门限，使总体、两域、每视频正确保留≥95%；原正式policy/原95%全帧覆盖点另报。不为各视频部署不同门限。

B必须：总体同接受数和精确同CR的FA均严格少于A/simple/score；各域/视频同数保守并列下限及同CR FA不增；最长连续正确FR不增；至少一个Real视频同CR严格减少simple FA，且该视频错误AUROC严格优于A及simple。仅同CR FA改善而视频内排序不改善不能通过新增机制门槛。并列无法精确匹配时记录不确定，不能用GT挑选并列成员；各组独立参照不可相加。

完整报告包含FA/FR/ED/CR/MISSING、输出/接受/正确全帧覆盖、正确保留及接受错误比例、分域逐视频AUROC、视频内宏平均与配对加权AUROC、pooled AUROC、连续正确FR、正式policy点。中心仅统计输出帧命中，另报输出覆盖与全帧中心正确覆盖。连续拒绝错误不计正确FR。TRAIN拟合不是泛化证明，VAL已反复开发暴露；TEST已多次暴露，仅冻结报告，不用于改合同、选权或门限。

失败状态VAL_FAILED_STOP：不采用、不反选A、不追加预算/改0.25/0.5/激活/容量，不自动进入TEST。全部通过才记VAL_PASS_FROZEN_TEST_PENDING，不自动替换正式policy。失败候选只记运行记录，不写主要成果。

新增统一入口`crane_project/tools/run_port_reliability_within_video_rank_v1.py`、独立review、协议/来源JSON、NumPy与Torch核心、测试；脚本`tools/run_port_reliability_within_video_rank_v1.sh`。后续获授权Git同步后，在项目根目录/mmrotljj执行`bash tools/run_port_reliability_within_video_rank_v1.sh RUN_ID`。脚本GPU3优先、其次2，两卡忙则停止；训练源码必须与HEAD/source SHA一致，避免结果绑定未提交代码，其他既有改动保留。该指令尚未执行。

日志和结果集中`work_dirs/port_reliability_within_video_rank_v1/RUN_ID`，候选result含配对清单、模型JSON、末态权重、逐帧风险、统计、训练梯度日志及独立审查。退出时仅压缩本RUN_ID到work_dirs根目录，不含.pth或旧特征/ROI缓存；本轮没有生成结果包或训练结果目录。

**本地实际验证：** 29项NumPy/来源/实际3445帧标签独立一致性/完整配对标量复算/评价与接口测试通过；9项Torch检查因本地无Torch明确跳过，服务器训练前必须全部通过。prepare验证2558 TRAIN＋887 VAL来源及缓存配对、32437配对支持、3444输出中性risk和3445帧原标志保持；无训练、无TEST。Python语法、shell语法和diff检查通过。实际GPU梯度、训练、保存重载及性能仍未执行，不能写成收益成立。

### 42.4 用户授权服务器运行、回传复算与实际结论（2026-10-09）

第42.1～42.3的未运行状态是实现时的历史记录；本轮用户明确授权“去服务器上启动训练，和之前流程一样”。代码提交推送347f32f，服务器原路径`/media/omnisky/personal_files/ljj/symEOOD`快进同步。GitHub专用代理失效，通过单次pull取消环境及GitHub-specific代理成功，不修改全局配置。GPU3忙，脚本选择物理GPU2，mmrotljj/Python3.8.20/Torch1.13.1+cu117/NumPy1.24.4。38项服务器测试、prepare及两步丢弃smoke全部通过；两臂各1000步，固定最终epoch100，实际训练评价耗时41.36s（非完整作业耗时、非检测延迟）。

**结论VAL_FAILED_STOP，不采用、不反选控制、不追加训练、不进入TEST。** 完整TRAIN2558＋VAL887的3445行、435项汇总、60项AUROC、32437配对均经双端独立复核。Torch/独立NumPy候选重放最大差2.22e-16。退出码0；原工作区/暂存指纹保持。来源/正式policy及框、score、中心、方向、输出身份保持。

| 完整VAL公平比较 | 候选BCE＋rank | BCE控制 | simple | score |
|---|---:|---:|---:|---:|
| 同850接受FA |191|194|197|197|
| 同850接受FR |4|7|10|10|
| 同CR659的FA |191|196|202|199|
| pooled错误AUROC |0.797239|0.786032|0.785626|0.684394|
| 视频内宏AUROC |0.558693|0.548778|0.549244|0.549183|

候选实际全局正确保留659/663、FA191/FR4/ED32/缺失1，接受850/887；两域/所有视频正确保留≥95%，最长连续正确FR未恶化。原正式policy另报，不将新工作点冒充正式门限。

| 分组独立同CR比较 | CR | 候选FA | BCE控制FA | simple FA | score FA |
|---|---:|---:|---:|---:|---:|
| Real |196|143|149|156|157|
| Sim/sim_seq10 |463|48|48|45|44|
| real_seq07 |60|130|128|130|144|
| real_seq14 |136|13|13|13|13|

分组控制门限独立计算，不可相加代表全局在线结果。seq07 AUROC0.591809较控制0.561434及simple0.539193提升，但同CR60的FA与simple相同、比控制多2；seq14 AUROC0.403281与控制相同、低于simple0.415158。Sim AUROC0.680990低于控制0.681618/simple0.693382，同CR FA比simple增加3。没有Real视频同时达到同CR simple FA严格下降及视频内AUROC严格增量，因此预先规定机制门槛失败，不修改合同。

**新增事实：** TRAIN视频内宏AUROC控制0.999666、候选0.998140，说明BCE已几乎完成TRAIN视频内排序；候选末态rank loss0.322887，控制0.307753。候选TRAIN/VAL绝对残差>0.49比例降到3.09%/0.11%（控制98.55%/83.52%），大幅缓解饱和但不形成稳定跨视频收益。VAL仍847风险降低、39升高。新增项确实改变了优化轨迹，不能把此次失败写成“没有梯度”或“仍全饱和”。

候选每epoch首批加权rank/BCE梯度范数比0.3191～1.4408，总梯度最大0.4631（控制0.3640），没有clip5裁剪；前后范数约保持，1e-16差异不记作裁剪。固定0.25未调参。32437配对仍来自71错误帧，Sim1错误，不能扩大为独立错误支持。

**推断与停止：** 本轮证明目标排序监督/饱和改善本身不足，尚不能指定唯一训练根因。建议封存该合同，停止在同缓存/头上调系数、门限、预算或激活；若再开方案需先明确新增主框质量证据。正式M/simple保持，可并行整理论文现有证据，不声称可靠性已整体改进。

结果集中`work_dirs/port_reliability_within_video_rank_v1/20261009_within_video_rank_v1`，独立分析`result/interpretation.md/json`及mac_review；代码绑定347f32f。回传包SHA`5b08ec2c29fe28e86df59f869cf707dc167dc929983dc92ac1fb9bdfce68fc38`，安全解包无权重；服务器final_heads.pth SHA`2afb0cca8bcb9f442c74648026b9fc2d669c948000fd494d8e4f86d0580a261a`保留。记录回传、Git同步与仅本次压缩包清理见同目录回执，失败候选仅写运行记录，不写正式主要结果。

## 43. 空间质量表示＋分轴连续监督有限对照（2026-10-09）

### 43.1 用户授权与固定合同

用户授权依据上一轮具体设计完成代码、检查、Git同步、服务器原目录训练、回传复算分析、记录与本次包清理。目标仍是固定M上的尺寸错误判断；不是修框/深度优化。第42节候选失败保持，不续训其权重。新版本`port_reliability_spatial_axis_v1`；正式M=B24＋sigma1.5/epoch03及simple/policy/框/score/输出数量/中心与方向身份保持。GT仅离线监督与评价，无输出不判断，不填GT框。TEST不读取；已反复暴露的VAL仍作开发和单一门限标定，不声称独立确认。

**待验证假设：** 可学习的局部空间质量表示以及有符号双轴连续监督能否形成跨视频增量。没有证明特征压缩、二值标签或支持失配为唯一根因。借鉴ReDC细粒度定位质量动机，但不实现原CAR/CR/DDE，不继承校准或概率保证。

重新在完整TRAIN/VAL最终保存M框位置提取冻结P3：256×9×9与1×9×9有效支持，按真实sx/sy映射、长短轴规范及pi周期采样，1.5上下文且源采样器最小16模型像素。重放B和sigma1.5确认来源，重放值不替换保存M。只提取原M，无合成框/跨尺度/排序辅助；插值不增加细节。TRAIN完成末态保存后才提取VAL特征。

两组卷积257→8(1×1)、8→8(3×3/pad1)，ReLU；展平648＋相对尺寸/比例2→16 ReLU，风险1输出/双轴辅助2输出，共13,115参数。三臂网络/初值/预算相同；A=`coarse_bce`仅将标准化的9×9特征均值池化3×3后固定重复回9×9（保留相同支持标志），B=`spatial_bce`原9×9，C=`spatial_axis`同B＋辅助项。A不是旧PCA头精确复现，同容量不声称同反向成本。冻结simple锚点logit＋0.5tanh(a)，风险末层0，精确中性expm1读出；辅助末层N(0,.001)、bias0，三臂deepcopy同状态。

离线监督：规范长/短边`e_j=(pred_j-GT_j)/GT_j`，`t_j=e_j/.1`；二值错误为任一绝对相对误差>.1。C=`平衡BCE＋.25×mean(SmoothL1(predicted_t,t,beta1))`，辅助项按真实帧/轴平均、不类别加权、不截标签。A/B辅助只记录、不反传，辅助参数无梯度、无衰减更新。最终尺寸标志只用风险门限，辅助误差绝不修框/直接生成接受标志或称正确概率。

完整TRAIN2558（71错误/2487正确，Sim1错误）；完整VAL887（886输出/663正确/223错误/1缺失）。TRAIN-only支持>0格点逐通道均值/标准差，标准化后无支持位置置0；两几何描述同用TRAIN矩。没有PCA、外部数据/新子角色。空间卷积/隐藏层FP32，风险及BCE/辅助loss FP64，TF32关闭、确定性；不是第42节全FP64头逐位复现。seed1701、100epochs/batch256、各1000步总3000；AdamW lr.001、仅weight wd1e-4、clip5。两步smoke丢弃后重置三臂同初值/空优化器，只评最终epoch100。

### 43.2 预检与验收

预检：长短边交换/有符号目标、3×3布局/细节抹除、支持掩码、TRAIN-only矩、中性风险/三标志保护、NumPy卷积标量参考、实际完整标签；Torch侧三臂相同初值与13,115参数、中性可微、C首步空间梯度、A/B第二步空间梯度且无辅助反传、辅助导数/有限差分、末态Torch/NumPy重放。每epoch首批分别记录BCE和.25辅助参数梯度范数/比值/余弦/分参数及统一裁剪缩放；每步记录总梯度裁剪前后，不能据loss数值断定系数最优。

VAL每方法一个全局整tie最小门限，使总体/每域/每视频正确保留≥95%。仅预定C可晋级：总体同接受数与精确同CR FA严格优于B/simple/score，各域/逐视频FA不增加，最长连续正确FR不增加；至少一个Real视频同CR simple FA严格下降且视频内AUROC严格高于B/simple。A仅机制控制，不能事后反选A/B；并列界保守，不GT拆tie。连续误差预测bias/MAE/P95分域逐视频另报，不以它们代替判断收益。完整FA/FR/ED/CR/MISSING、接受/输出/正确全帧覆盖、中心仅输出命中与另外两覆盖、pooled/视频内AUROC、原policy点均保留。

失败即VAL_FAILED_STOP，不现场改系数/预算/容量/激活/门限，不自动TEST；通过只记VAL_PASS_FROZEN_TEST_PENDING，不自动替换正式policy。失败实验只写运行记录，不追加论文主要成果。

### 43.3 实现、本地检查与运行安排

新增核心/torch、runner/reviewer、协议/65文件来源清单、两份测试及脚本。实际本地15项数值/接口检查通过，10项Torch因为缺库跳过，须服务器实际全部通过。本地完整TRAIN标签与prepare身份/中性标志核对通过；本地没有原B/midpoint权重和原ROI清单，`--allow-missing-server-inputs`仅本地prepare放行并明确`all_original_inputs_checked=false`，不是来源已全部核验声明。服务器正式训练不可带此参数，全部原输入pin/HEAD源码强制核验；本机结果review凭服务器原输入核验回执补足缺少原权重的来源链，不下载大权重。

运行统一目录`work_dirs/port_reliability_spatial_axis_v1/RUN_ID`。脚本GPU3优先、其次2（显存<200MiB且util<5%），服务器原项目与mmrotljj，保留工作区/暂存指纹。包在work_dirs根目录，只含本RUN_ID，排除.pth；本项新原始空间NPZ保留并回传用于全3445行独立卷积重放，不带历史缓存。双端结果/权重保留后仅清理本次包。

此处为运行前合同，尚无本候选性能结果；实际服务器运行与分析在43.4追加。运行入口：`bash tools/run_port_reliability_spatial_axis_v1.sh RUN_ID`。

**首次服务器工程停止（保留）：** 源码4c304a1，RUN_ID=20261009_spatial_axis_v1；25项检查及全输入哈希通过，建立冻结检测器后、第一帧特征提取之前因迁移采集代码的`historical.B_PATH`层级引用错误停止，训练0更新、VAL未读取。修正为`historical.historical.B_PATH`并补来源引用检查，合同/模型/数据/预算不变；保留failure.json/日志，不覆盖旧目录。首次包SHA6be11c…保留到最终收尾时精确清理。修正版另RUN_ID执行，不将此次工程失败写成候选性能失败。

**第二次工程停止（保留）：** 修复来源3a8fc1b，RUN_ID=20261009_spatial_axis_v1_fix1；26项检查与来源预检通过，在第一帧B/M重放后的尺寸标签保护处发现原采集函数迁移遗漏`ab`名称，训练仍0更新、VAL未读。改用相同定义的metrics.bad，并用符号表检查采集函数所有全局引用均可解析；不改变标签合同或损失。第二次包SHA8afa1a…，旧目录/failure.json保留，下一次以fix2目录执行。

### 43.4 修正版实际运行、双端复算与结论（2026-10-09）

训练源码d3e2f817c9f3ea53e1c42f0da398f67d11d6febf，RUN_ID=20261009_spatial_axis_v1_fix2；原服务器目录/mmrotljj，物理GPU2，GPU3已有作业保持。27项服务器测试全部通过，原输入哈希全核验；两臂工程失败均0更新/无VAL，修正版按原合同重新同初始化，三臂各100epoch/1000更新，总3000，smoke丢弃，不选中间权重。更新阶段29.66s，report阶段982.77s（不含后续review/打包，非在线延迟）。B/M六字段重放差全0，正式框/score/中心/方向/输出/原policy保持。

**状态VAL_FAILED_STOP，不采用、不反选A/B、不追加预算或调系数/门限、不进入TEST。** 双端独立复算3445行、690项状态/同数/同CR汇总、75项AUROC、门限与失败门槛一致。独立卷积风险重放约4.65e-7、辅助输出8.78e-7，属于预定FP32空间运算重放误差；原风险中性精确、保存重载精确。A/B/C各13,115架构参数；A/B辅助头未反传，实际更新13,081，C更新13,115，不混淆总参数与实际梯度参数。

| 完整VAL公平比较 | A粗空间BCE | B空间BCE | C空间＋轴监督 | simple | score |
|---|---:|---:|---:|---:|---:|
| 同858接受FA |210|202|201|201|199|
| 同858接受FR |15|7|6|6|4|
| 同CR657 FA |221|202|201|200|199|
| pooled错误AUROC |0.772410|0.783522|0.784192|0.785626|0.684394|

三新臂实际工作点恰均CR657：A/B/C实际FA221/202/201，说明完整空间相对粗控制有信号，连续监督再少1FA却仍未超过simple。C实际FA201/FR6/ED22/CR657/MISSING1，各域/视频正确保留≥95%，正确连续FR保护通过；主要失败是同CR错误接受，不能归为连续拒错误算FR。

| 分组独立同CR | CR | C FA | B FA | simple FA | score FA |
|---|---:|---:|---:|---:|---:|
| Real |193|155|156|154|154|
| Sim/sim_seq10 |464|46|45|45|44|
| seq07 |63|143|141|141|150|
| seq14 |130|12|13|13|13|

这些分组门限独立，不可相加代表全局在线规则。seq14有局部1FA收益，seq07/Sim退化，正式原policy点在report另报，不用新门限覆盖旧结果。

**新事实：** C的seq07全部225输出、Sim全部512输出修正绝对值>.49，分别211/471在1e-12内为-.5；这两视频风险均降低，但视频内AUROC与simple完全相同0.539193/0.693382。C完整VAL饱和98.87%，B99.21%；seq14 C AUROC0.479072比B0.404412/simple0.415158高，不能据此宣布逐视频普遍改善。第42节缓解饱和仍失败，本轮也不能将饱和视为唯一根因并临时改.5。

C辅助残差估计相对固定零误差预测参照：TRAIN总体长/短MAE3.045/2.784→2.584/2.688百分比点；VAL总体5.967/6.059→7.347/6.655，Sim4.322/4.345→6.739/5.317，seq07 10.645/12.013→10.819/12.172，seq14 4.558/2.959→4.191/2.923。这里是误差估计的准确性，绝不是检测框被修正后的几何误差；A/B辅助头未训练，不把与其比较称公平回归收益。C TRAIN视频内宏AUROC0.968885，VAL0.570549，连续监督没有稳定迁移。

C100次分项梯度记录加权辅助/BCE范数比.0512～54.5073（首批.6841），辅助首步空间梯度非零；高比值可能来自BCE接近0，不能据此宣称.25错误。C/B最大总梯度3.478/2.604，均未clip5；A3次裁剪、最大6.441。没有把辅助梯度或裁剪当作失败遮掩。

**决定与边界：** 封存本合同，保留正式M/simple，尚不能隔离P3、受限读出、标签/真实错误支持的唯一根因，也不构成原ReDC复现失败。先整理正式可靠性成果及负结果边界，不自动展开新训练。中心仅输出命中874/886、输出覆盖886/887、全帧正确覆盖874/887保持。

完整结果集中work_dirs/port_reliability_spatial_axis_v1/20261009_spatial_axis_v1_fix2；独立interpretation.md/json与mac_review在result。新空间特征保留两端、末态.pth保留服务器SHA621fd91dc62f006323f676291f982cbd55472a1023bd0351735a0f2aeb232f12。最终回传包SHA9e2850283a2858ec26ebe8ed75ee7bb51d194a42a189209beb2fbe1c37906f95，安全成员/路径核验通过。首次和fix1的零更新失败日志已取回保留。服务器既有工作区/暂存指纹保持b48cc0…/d54105…。回传、记录Git同步与仅本项包/bundle精确清理回执保存同result，不写论文主要成果。

## 44. 按Real序列排除的训练预测来源对照（2026-10-10）

### 44.1 授权、假设与固定范围

用户授权按“来源合同→资源预检→两套辅助训练→折外预测→simple对照→完整VAL”执行，并沿用本地修改/Git同步/服务器原目录与mmrotljj/后两张空闲卡/回传独立复核/本项压缩包清理流程。正式M=B24＋sigma1.5/epoch03、原simple/policy、三分量接口、几何/深度公式保持。仅检验改变判断器训练预测来源是否能提高正式M上的尺寸错误辨识；未宣称错误支持是唯一根因，不增加新质量网络或特征。

### 44.2 数据与初始化合同

完整TRAIN Real1810、Sim748；Real A=seq01/05共899，B=seq06/12/13共911。辅助A用B+Sim1659训练并预测A899，辅助B用A+Sim1647训练并预测B911。各保留当前SymEOOD ResNet50/FPN、SymNFL/SymKLD/SymPOLA与B增强，初始化只用通用ImageNet ResNet50；禁止加载B24、原SymEOOD港口权重及正式midpoint。各辅助B固定24轮末态，midpoint sigma1.5从中性初始化只读该折允许数据1/.5缓存，固定3轮末态；原域均衡batch8/lr.001/clip10目标不改。无辅助VAL选权、无TEST。

seq12两个连续源片段绑定同组；seq13有独立源文件记录。旧seq01/05/06缺完整采集事件关联证据，故仅称整序列排除、不称独立采集事件或整域折外；TRAIN跨A/B精确图片SHA重复为0。Sim只有seq08，保持正式M原in-sample材料，明确不是Sim折外。原图/标注通过manifest逐文件SHA核对；新训练目录只建允许身份的符号链接，原数据不复制、不改动。

### 44.3 预检、判断器与停止条件

资源预检独立进程执行8个实际FP32检测训练step，batch2、1024输入；记录loss/梯度、每步耗时及显存allocated/reserved峰值，预检模型丢弃，峰值reserved须<物理总显存95%，有限数值；不自动改batch。服务器已缓存通用ResNet50，不需要港口任务权重初始化。

OOF标签由各自实际预测与GT计算双边最大相对误差>10%，漏检保留MISSING、不标成尺寸错。按视频检查辅助输出覆盖≥95%、输出中心<15px比例≥90%，属于工程失效护栏，不保证错误类型代表性。若不通过，标AUXILIARY_SOURCE_FAILED_STOP，正式VAL评分不启动；不因此加训练或重分组。

判断器两主臂同simple3特征、float64平衡logistic、L2=.1/Newton100/梯度1e-8：in_sample复现原TRAIN拟合，oof使用Real折外+相同Sim原预测；另列common_in_sample/common_oof共同有框身份对照，只有oof可晋级，不反选共同子集臂。先固定四臂TRAIN模型，再用同一正式M完整VAL887帧/886输出校准各自单一全局门限，分域/逐视频正确保留≥95%，whole ties。报告FA/FR/ED/CR/MISSING、同数/同CR对照、逐视频AUROC、连续正确FR/错误FA与中心输出/全帧覆盖。oof总体同数/同CR须严格优于in_sample/原simple/score，各分组FA不增且连续正确FR保护；失败VAL_FAILED_STOP，不重选辅助权重/特征/分组/门限。通过只标VAL_PASS_FROZEN_TEST_PENDING，未自动部署或访问TEST。

### 44.4 实现与运行状态

新增独立core/runner/CPU reviewer/protocol/source pins/tests/shell入口。训练全过程原M/policy只校验来源，不更新；候选policy单独保存，在线只需要正式M及一个simple，不加载辅助模型，不读GT/sequence/domain。深度使用价值未被本实验建立。所有产物集中work_dirs/port_reliability_oof_v1/RUN_ID/result，两套辅助权重分别位于A/B；分析包排除权重、缓存及符号链接数据目录。本段为初始实现状态；实际预检、兼容修复及训练启动进展见44.5/44.6。

### 44.5 首次资源预检与零正式更新兼容修复

训练源码ccab8a1，首次RUN_ID=20261010_real_oof_v1，GPU3。14项服务器检查通过；8步资源预检实际完成，模型31,560,792参数，FP32/batch2/1024，显存峰值allocated2,589,477,376 bytes（约2.41GiB），reserved2,822,766,592 bytes（约2.63GiB），8GB GTX1080资源护栏通过。第1步含启动/内核开销13.70s，后7步约0.96～0.99s/步；这不是完整训练或部署延迟。按约39,700个两辅助B训练更新粗估约11小时，另加缓存/head/推理与IO，非完成ETA保证。

进入实际tools/train.py后，MMRotate compat_cfg发现data.samples_per_gpu与data.train_dataloader.samples_per_gpu重复设置，零正式训练更新即停止，退出1；首次包SHA e754e0f5d674625ea302f8336197d150e8ee57f7c64b8ab24de08f0c6aabc187，原失败日志保留。修复仅移除新旧loader重复字段并在资源预检前执行实际compat_cfg；增加服务器配置兼容回归，不改架构/数据/预算/初始化/门槛，不把此工程失败计为性能失败。修正版使用独立RUN_ID，资源预检模型仍丢弃。

继续静态核对compat_cfg发现其在validate=False时仍读取data.val/test字段；辅助配置改为无路径空占位，防止框架字段访问错误，实际不会构造原VAL/TEST数据集。该问题在修正版启动前修正，无新增训练失败。来源闭包追加compat_config.py，共81份。

### 44.6 修正版真实训练已启动，性能结果待完成

修正版训练源码7b3f2a6，RUN_ID=20261010_real_oof_v1_fix1；服务器原项目目录/mmrotljj，物理GPU3（逻辑0），后台脚本PID1497581，实际tools/train.py PID1500331。15项服务器检查全部通过；修正版8步预检通过，参数量及显存峰值与首次一致，后7步耗时中位数0.9867s。预检模型丢弃后重新按合同初始化，没有从首次失败或预检状态继续训练。

2026-10-10 10:09:53服务器日志确认辅助A正式训练已达到Epoch[1][150/830]，batch2、最大24轮、workflow仅train、--no-validate、seed1701。允许Real911/Sim748加载身份与合同一致。第100/150步记录loss1.9031/1.4762，显存日志2469MiB；这些是训练运行/有限数值证据，不是判断准确性或泛化收益。两套辅助检测器总训练量约39,700步，按资源预检粗估约11小时，另加midpoint、折外推理与IO，不作为完成时间保证。

后台流程依次执行A24轮末态→A自己的sigma1.5 midpoint3轮→A排除序列预测→B相同步骤→来源护栏→TRAIN四组固定simple拟合→完整VAL→服务器独立review与本RUN_ID打包；源码中实际完整VAL评分仍在TRAIN模型冻结及来源护栏通过之后。来源或VAL失败均按原合同停止，不自动TEST或替换正式policy。当前B、各midpoint、OOF、判断器与VAL尚未完成，不能宣称可靠性改进。最终结果回传、双端复算和修正版包清理也尚未发生，不将“后台已启动”写成全部流程完成。

首次零正式更新失败包已回传，SHA与安全成员检查通过，完整失败目录/日志/预检记录保留两端；仅该次tar.gz已精确删除，清理回执位于首次运行目录。修正版完成后包将位于服务器work_dirs根目录`port_reliability_oof_v1_20261010_real_oof_v1_fix1.tar.gz`，权重不回传。服务器既有工作区/暂存指纹b48cc0…/d54105…保持；正式M、原policy及历史结果身份保持。

### 44.7 实际结束、故障核对与修复（2026-10-10）

**实际未进入VAL，不能报告本候选VAL性能。** 原fix1退出1，failure.json记录辅助B的midpoint没有可用域支持。两个辅助检测器确已完成24轮：A19,920更新，B19,776；A自己的midpoint3轮/1308更新并生成899帧折外结果，B在midpoint首步前停止。原错误/权重/日志保持，不将代码检查通过写成训练性能通过。

| A排除序列预测 | 帧数 | 有框 | 尺寸正确/错误 | 输出中心<15px | 输出覆盖 |
|---|---:|---:|---:|---:|---:|
| seq01 |339|103|2/101|37/103|30.38%|
| seq05 |560|500|9/491|498/500|89.29%|
| 总体 |899|603|11/592|535/603|67.07%|

A不满足既定逐视频输出覆盖95%；seq01输出中心命中也不满足90%。这是折外检测来源失效，尚不是判断器辨识结果。正式M在同899张TRAIN图上899有框/854尺寸正确，但M已训练这些图，两者不构成检测泛化能力的公平优劣比较。

GPU3只读重放B完整允许TRAIN的1/.5尺度共3294视图：Real每尺度899、Sim每尺度748，四组输出与eligible均0。额外核对标准init_detector与辅助model_for的加载参数逐位相同，三个代表帧的标准预测也完全一致；最高主头sigmoid分数0.03318/0.04124/0.03348，低于冻结0.05输出门槛。此前9帧抽查权重有限，无NaN。故无可用样本不是简单加载接口或中心门槛错误；未证明训练失效的唯一优化根因。未降score阈值、放宽匹配、填GT框、选择其他轮次或重启训练。

用户要求修复并完成训练/VAL后，修复提交a9b0e37/c87a8b6：空域支持先保存逐域/尺度统计并返回AUXILIARY_SOURCE_FAILED_STOP；每折立即应用原视频来源护栏，A不通过就不继续消耗B预算；单折不额外要求两质量类别，完整TRAIN仍要求两类。补正候选size门限的VAL真值标定身份，导出policy逐帧在线重放/篡改检查，以及JSON长度直方图整数键与字符串键兼容。没有修改模型/特征/预算/工作点/正式policy。代码修复解决工程诊断与复核问题，不修复已经失效的辅助权重，因此原合同下没有可继续拟合并进入VAL的合格完整OOF来源。

本地20项测试19通过/1服务器环境跳过，服务器20项全部通过。新增端到端测试用**原M预测作为合成OOF**，验证拟合→完整VAL计算→导出→CPU复核及篡改拒绝；这些测试临时结果自动清理，不是辅助模型的新VAL实验或性能证据。实际A899行本地独立重算帧数/二值尺寸标签/中心命中与服务器一致；B3294视图由服务器GPU重放，本机仅核验回执，不声称Mac独立GPU复现。

结果集中原result/postmortem，包含B_training_support.json、source_failure_analysis.json、repair_tests.log与mac_review.json。原回传包SHA9854b30a84269169b819bf3588d3b99372940629d9a01cfbdc5e445b588704a6已核验并安全解包；不下载权重。服务器GitHub访问失败，使用校验后的Git bundle快进同步到c87a8b6，原工作区/暂存指纹b48cc0…/d54105…保持。本次仅来源诊断与工程修复，TEST未读、VAL候选未测，不写论文主要成果。

### 44.8 辅助训练实现修复与重训边界（2026-10-10）

**授权与状态。** 用户要求先修复已查明的训练实现问题，并询问是否需要全部重训。本轮完成本地代码、受影响检查和服务器一次性工程验证；未提交/推送源码，未覆盖服务器源码，未启动新24轮训练或候选VAL/TEST。正式M、policy、历史权重及失败结果保持。新训练修订身份为`normalized_padding_per_image_gt_strict_v2`，旧prepare/配置及resume状态不能进入新流程。

**已确认问题与尚未确认根因。** 辅助B保存的50步日志聚合显示第4/6/8/24轮主分类loss均值约.134/.446/51.558/.340，主回归约.124/2.169/3.787/.688，裁剪前梯度约5.34/162.27/5202.73/77.63；中期失稳已发生在其自己的TRAIN，不能只归因于未见视频。训练配置在Normalize之后Pad114，服务器实际有效像素约−2.12～2.64，而填充114，抽查一帧填充占比84.45%。SymNFL原实现合并batch的预测与GT做空间权重，存在跨图匹配。失稳加重接近O2M收缩到O2O阶段，但没有隔离对照证明分配调度、padding或跨图权重是唯一根因。

**修复范围。** 原B配置字节不改；仅OOF的`prepare`通过`port_reliability_oof_training_v2.repair_config`启用：①将原始RGB114按各Normalize的mean/std转为归一化三通道填充值；②主头传递预测及正GT的图像ID，SymNFL逐图匹配、逐图top-K，不使用其他图GT，空GT图保留普通Focal权重；③主/ATSS辅助loss、SymNFL/SymKLD/SymPOLA距离及实际参数梯度严格有限检查，异常在optimizer更新前停止；④记录真实正样本总数/逐图最小最大、分配迭代、裁剪前后范数和比例。共享类的新增开关默认关闭，checkpoint状态键不增加，历史配置/推理入口保留。

**计算范围披露。** 空间top-K300现在是每图每层，batch2最多600个；旧批次合并版本每层最多300个。此为逐图隔离修复的一部分，不声称与旧版本相同FLOPs，也不把联合修复解释为单因素消融。架构、24/3轮、seed1701/1703、SGD/lr及clip10、辅助TRAIN排除分组、score门槛.05和来源95%/90%护栏均不改。新来源清单84份。

**工程验证（非性能收益）。** 本机纯配置检查5项通过；新测试实际Torch/MMCV部分13项只能在服务器核验。服务器mmrotljj中将当前修改源码仅加载到内存，18项新检查＋20项原OOF检查共38项全部通过/无跳过，包含逐图GT/top-K隔离、单图旧新值与logit梯度一致、空GT、真实主头标签/梯度路径、真实Normalize/Pad、异常梯度不得更新参数和历史状态键/默认配置保护。首次检查有2个测试夹具把GT标签与metadata参数顺序传反，修正测试后以上38项通过；未把该夹具错误当作训练根因。

GPU3空闲时以辅助A原允许TRAIN数据执行8步可丢弃检查：新鲜ImageNet初始化、FP32/batch2/1024、SGD.0025/clip10，使用新增严格optimizer hook；8步每图9正样本，总18，主分类loss约.356～.481，主回归约.881～1.542，所有异常计数0。裁剪前范数约80.44～251.12，裁剪后约10，reserved峰值2,822,766,592 bytes（约2.63GiB），资源护栏通过。模型/优化器随进程丢弃、没有保存权重；这些数据不能证明24轮后稳定或折外来源通过。原端到端单测使用合成OOF重放保存VAL，仅证明工程接口，不是新的候选VAL性能。记录集中`work_dirs/port_reliability_oof_v1/training_repair_v2_validation/validation_report.json`及同目录日志。

**是否重训。** 不重训正式B24＋sigma1.5/ep03、现有simple或历史实验。若继续本项，两套辅助A/B均需按修复后的配置从通用ImageNet重新训练固定24轮，再分别重建自己的中性midpoint3轮及折外预测；禁止在旧权重/optimizer上续训，否则混合两种训练合同。原判断器本次尚未拟合，新完整OOF通过来源护栏后才首次拟合并开展完整VAL。依旧每折先验收，A失败立即停止，不自动消耗B预算；新运行必须用独立RUN_ID、保留旧失败证据。当前修复仅支持新一轮受控验证，尚未证明修复必然提高可靠性。

### 44.9 GPU3重训授权与显存稳定护栏（2026-10-11）

用户授权A/B均按修复后合同重新训练，物理第四卡GPU3；启动前GPU2与GPU3都必须无人使用。2026-10-11 07:52:52 CST只读快照两卡各12MiB/0%，无计算进程。旧失败结果、正式M/policy及服务器既有暂存内容保留。当前先准备源码同步及启动前检查，不能据此写成正式训练完成。

用户明确允许启动预热，要求其后显存稳定不上涨。新增CPU标量显存护栏：每阶段32步预热，随后reserved不得超过预热峰值（容差0），allocated仅允许16MiB临时张量波动；全程固定4GiB上限。辅助检测实际optimizer、缓存生成、自己的midpoint更新和折外推理分别检查，任何超限抛错、退出本作业；不得自动续训或改batch。缓存张量在no_grad内提取后转CPU，禁止按帧积累GPU特征。原8步资源预检扩充64步可丢弃显存检查（共72步），不保存权重，不计入固定24/3轮性能预算，也不接触候选VAL/TEST。

启动脚本只选物理GPU3，不回退到GPU2；测试前和进入GPU流程前检查两卡利用率、显存与计算进程。启动时的空闲快照不能保证其他人以后不提交作业，不中止或修改他人的进程。代码检查和真实显存结果、启动PID及RUN_ID将在同节继续记录。

第一次启动20261011_real_oof_v2_gpu3：服务器20＋22项实际测试全部通过/无跳过。单测后第二次空闲检查显示GPU3无计算进程、12MiB但NVML上一采样区间利用率3%，按严格0%条件退出3；未进入prepare/preflight/训练。随后快照GPU2/3均12MiB/0%。补充单测后3秒冷却再检查，不降低占用条件；下一启动另用RUN_ID，保留此次拒绝事实。

**实际放行与后台状态。** 代码修复a0d9f9f，冷却检查修复1222463；GitHub已推送，服务器代理127.0.0.1:17890不可用，使用SHA校验后的Git增量包快进到1222463。服务器原工作区/暂存指纹b48cc0…/d54105…均保持。新RUN_ID=`20261011_real_oof_v2_gpu3_fix1`，后台launcher PID1032248、实际A训练PID1043715；cwd为指定项目，mmrotljj，物理GPU3，进程忽略SIGHUP且stdin=/dev/null。日志集中该RUN_ID目录，未覆盖fix1历史失败或首次空闲检查退出目录。

新运行服务器20＋22项检查全部通过。72步丢弃预检约70.464秒，通过；第33～72步40次采样allocated均382,161,408字节，reserved均2,822,766,592字节，范围上限等于下限。峰值allocated2,589,299,200，reserved2,822,766,592字节（2692MiB）；nvidia-smi包含CUDA驱动上下文约3012MiB。预检权重丢弃，无VAL/TEST张量推理。

2026-10-11 08:01:52 CST正式辅助A到Epoch1 step50/830，来源Real911＋Sim748，main正样本每图9、全部异常计数0；已越过32步预热并通过显存护栏。主分类.4426、主回归1.2503、裁剪后梯度约10均为此初期日志，不是性能通过。第三卡GPU2仍12MiB/0%，仅GPU3约3012MiB/100%用于本实验。当前只完成启动/初期验证，A/B24轮、自己的midpoint3轮、来源验收及候选VAL尚未完成；A来源失败或任何显存增长仍按合同停止。不能保证未来其他人不会新占卡，也不把短预检视为全程显存证明。

### 44.10 辅助A来源失败、推理遗漏及有限学习检查（2026-10-11）

**已经确认的事实。** `20261011_real_oof_v2_gpu3_fix1`的辅助A完成24轮/19,920更新，随后midpoint训练支持两域均为空，返回`AUXILIARY_SOURCE_FAILED_STOP`；未训练自己的midpoint或辅助B，未拟合判断器或开展候选VAL/TEST。受控停止的shell exit 0不是性能通过。正式M/policy及旧失败权重保持。

只读核对发现前次修复只修改辅助TRAIN的归一化灰色填充，`cache_views`和折外推理却仍复用历史B的Pad114，造成辅助训练/推理不一致。这是实现遗漏；原B配置必须保留，不得全局改成新填充。内存修正辅助推理后，用A24重放自身全部允许TRAIN：Real911、Sim748，每域分别1/.5尺度，共3318视图；全部有输出，八个序列×尺度组中心<15px均为0。GT中心与数据集标注、native原图坐标还原最大差均0；标准Sim无填充也失败。12帧抽查权重逐项一致、native与封装预测一致；主头及同模型ATSS头中心命中均0/12，全部主头候选中没有同时满足中心及双边尺寸条件的框。故不能把填充遗漏或top-1错选宣布为唯一原因。

损失/梯度有限与显存稳定不等于学会检测；保存的50步日志聚合在第8/10轮出现大幅波动，唯一训练根因尚未隔离。此前只读诊断GPU reserved约488MiB，已退出并释放至12MiB；没有改权重、门限或重训。

**本次授权与固定检查。** 用户认可统一辅助预处理并进行短程学习检查。本轮仅修复辅助专用推理模板复制：同时供自身TRAIN缓存与折外推理使用，校验mean/std/RGB/padding size与实际辅助TRAIN一致；不修改正式B/M入口。新增独立诊断脚本从失败A允许TRAIN中按身份确定一张Real（首序列排序中位）和一张Sim（排序中位），不按预测/GT质量选样。关闭随机增强，保持1024/FP32/batch2、原损失、SGD、线性warmup和clip10，ImageNet新鲜初始化重复固定批次256步；读出固定0/64/128/256步，不能加预算或选最佳步。失败A24仅作只读参照，诊断权重与优化器不保存、不供正式训练续用。

实际检查训练/推理像素张量逐位一致、GT中心还原、分配后编码GT解码与本图GT一致、正样本及裁剪前后/各模块梯度、clamp前正框饱和计数和最终预测。日志只存CPU标量；仍用32步预热后显存平台及4GiB护栏，先确认GPU2/3无人使用，只用GPU3。最终两张图均中心<15px且双边≤10%才说明本固定微批可学习；通过也不自动授权/启动A/B长训练，不推论泛化或可靠性收益。没有候选VAL/TEST张量读入/评分；原来源审计仍校验既有TRAIN/VAL证据身份。真实运行状态及结果待补。

**实际执行与复核。** 修复提交`280c298`已获用户针对`github.com/lusirs123/mmrotate-crane1`的明确推送授权并推送；服务器代理17890不可用，Git增量包SHA`af6f5b9f8ddad2bf46a6b81805abb5524dc321104c99225cc0b2916d16f2fd8f`验证后快进同步。服务器既有工作区/暂存指纹b48cc0…/d54105…保持。本机13项配置/新合同检查通过，原OOF20项19通过/1因无MMRotate跳过，另14项Torch检查在Mac跳过；服务器20＋24＋3共47项全部通过/无跳过。合成OOF接口单测不是新的候选VAL性能。

RUN_ID=`20261011_fixed_TRAIN_learning_v1`，指定项目/mmrotljj/GPU3后台实际完成256更新并退出0，GPU2未使用。按身份选中`real_seq06_00234`与`sim_seq08_00374`；新鲜ImageNet初始化重复同一固定批次，没有复用失败A/正式M。原A24在两图上中心误差336.856/422.056px；新鲜未训练为247.807/486.833px。固定64、128步两图中心和双边均合格，但最终256步如下，不反选早期权重：

| 最终固定微批 | 中心误差px | 长边相对误差 | 短边相对误差 | 双边≤10% |
|---|---:|---:|---:|---|
| Real seq06_00234 |10.193|10.167%|8.984%|否|
| Sim seq08_00374 |13.256|2.550%|6.477%|是|

**判定与边界。** 最终微批条件未满足（Real长边轻微超限），`tiny_batch_learned=false`；机械完成退出0不是条件通过。不加预算、不选64/128步、不启动A/B24轮、中点或判断器/VAL/TEST；诊断权重和optimizer丢弃，原失败checkpoint SHA保持。此轻微超限不能与原A全TRAIN零中心命中等同；早期两个固定样本能得到正确框，证明本检查的监督链可学习，但未证明完整数据/随机增强/后期O2M→O2O收缩稳定。本256步仍在O2M早期及LR warmup内，没有隔离完整训练崩塌的唯一根因。

TRAIN/inference像素逐位一致，GT中心还原最大差0，分配GT编码解码最大差7.629e−6；每图正样本跨FPN层汇总9，总18。主分类、主回归、backbone、neck、ATSS每一步均有非零有限梯度，256步全局clip10均触发，裁剪前范数48.126～244.380、中位梯度保留比例约.06239；正框clamp前饱和计数0。这些排除了两图早期的断梯度/错GT/尺寸clamp饱和，不能排除旧A24后期相同问题。总loss5.58684→.161083，不作为实际尺寸合格的替代。第33～256步224个采样allocated均432,494,592 bytes，reserved均2,845,835,264 bytes（2714MiB），范围均0；NVML约3034MiB稳定。阶段结束两卡均12MiB/0%，无本次计算进程。

本地安全解包与CPU独立复算10条固定读出、256更新/跨层正样本汇总/loss四项求和/显存平台，与服务器一致；Mac未独立重放GPU预测。首轮CPU复核错误假设每图只有一个FPN条目，已改为按图跨层汇总，未修改训练数据或重新运行。回执集中同RUN_ID/result/mac_review.json，原日志/合同/逐步明细保留；分析包SHA`70900742401b3f7bbf2347be5de5427f7f6272a7e705582a6fa8e9e9557c0a0a`校验成功。本次压缩包与临时Git包在复核后精确清理，不清理旧权重或完整结果。

**下一步建议，尚未执行。** 暂不重新花24轮预算；围绕“早期可学习、完整训练末态来源失效”定位长期失稳。优先在失败A24的固定TRAIN图上只读检查GT正样本分配、主/ATSS损失的实际梯度及解码/损失饱和，与已有早期日志并列；需披露分配/损失调度的迭代上下文不能从checkpoint自然恢复。没有证据前不调门限、取消clip、换网络或把失败归给可靠性判断器。最终主线仍是获得合格OOF错误监督，改善固定M的尺寸可靠性判断，不修改M框或重启深度优化。

### 44.11 完整训练失稳的日志与梯度取证（2026-10-11）

**范围与状态。** 用户要求定位完整训练中途失稳。本轮复用失败辅助A `20261011_real_oof_v2_gpu3_fix1`的全部训练日志与A24权重，只在其允许TRAIN上做无optimizer更新的诊断；没有改算法/损失系数/门限，没有重训、自己的midpoint、辅助B、判断器或候选VAL/TEST。正式M、policy、历史结果及另一几何窗口已有改动保留。证据集中 `work_dirs/port_reliability_oof_v1/20261011_training_instability_forensic_v1/`，包含脚本、原CPU日志、日志聚合、梯度/数值/通道报告及Mac复核；未下载权重、未新增压缩包。

**完整日志事实，不能当单步最大值。** 24轮/19,920更新，共384条TRAIN日志，每条为50更新聚合，epoch均值为这16个已记录窗口的均值，不含各轮未单列的最后30步。第2～4轮主分类/回归与裁剪前梯度较低；O2M的每图9正样本收缩至O2O的每图1在约4,000步前完成。第7轮及第8轮前600步窗口总体仍平稳，第8轮601～650步（全局6,411～6,460）首个明显尖峰：分类loss18.52031、回归2.25035、裁剪前梯度12,575.91404；此前50步为.29343/.33881/19.20439。更严重窗口在第11轮50步与第12轮300步：分类loss3,173.44182/7,599.30699，裁剪前梯度283,974.67971/287,351.55051。后者回归loss13.30395。最剧烈阶段主分类loss占主导，不能仅检查第8/10轮就认为已掌握全程。上述数值均有限，原NaN/Inf与显存检查不会拒绝这种巨大但有限的更新；clip10限制了当步总梯度范数，没有保证学会检测或恢复来源质量。窗口平均裁剪比例不是10除以窗口平均裁剪前范数，不能混用。

明显失稳出现时LR仍.0025，既不在1,000步温度退火/LR warmup结束瞬间，也不在O2M收缩结束瞬间。第17、23轮LR下降后loss缓和，末态来源仍失效。因此分配稀疏、归一化、LR、增强/样本及共享特征的作用仍需隔离，不能将“切换到O2O”宣布为唯一根因。

**失败A24的固定TRAIN梯度分解。** 按上轮相同身份取Real seq06_00234和Sim seq08_00374，关闭随机增强，1024/FP32/batch2，保持失败权重，不使用优化器。显式设置分配迭代19,920；SymNFL从checkpoint恢复的注册buffer为99,600，恰等于19,920×5层调用。原44.10“调度不能自然恢复”应精确理解为：分配器Python计数器不在state_dict，须显式补齐；SymNFL buffer已经保存，不是两者都丢失。

| 固定批次分量 | loss | 裁剪前参数梯度L2范数 | 共享backbone / FPN范数 |
|---|---:|---:|---:|
| 主分类 |.323277|5.677099|.233489 / 1.549123|
| 主回归 |1.096323|383.275301|73.722574 / 148.581738|
| ATSS合计 |.014832|.326786|.030129 / .166854|
| 总计 |1.434432|383.216183|73.647264 / 148.365982|

这是**末态固定两图**的实测，不是历史首次尖峰的分解，也不代表全TRAIN均值。主回归梯度约为主分类的67.51倍。按原全局clip10计算，本批梯度缩放约.026095，分类/辅助梯度同时缩小；共享特征上的主分类与回归方向余弦约−.154263。输出层按3 anchors×5自由度正确聚合后，编码(dx,dy,dw,dh,da)权重梯度范数约55.139/13.138/58.376/332.068/49.942；dh连同bias贡献回归输出层梯度平方的92.365%。这是**编码dh**，宽高交换后不能直接等同规范短边；也不能凭不同物理单位的叶子梯度大小判定角度是根因。直接物理框叶子KLD导数有限，而通过解码与网络后的参数梯度显著增大，说明仅看loss数值/有限性无法看见失衡。

**分配反事实与排除范围。** 同权重/同图/SymNFL tau=1下，只将分配上下文设为1,500步，恢复每图9个正样本：主分类/回归梯度11.480498/407.513833，总梯度407.990075，回归仍大；背景分类loss从.0416705降至.0046271（近9倍），符合正样本归一化变化，但正分类贡献仍占该末态批次分类loss主体。故恢复正样本数没有消除末态回归失衡；也不能据此追认历史O2O无影响。两种上下文均正GT中心解码最大差0，所选正框不触发中心/尺寸clamp或KLD loss硬上限；这些只排除本两图当前状态的断梯度/错GT/硬饱和，不排除历史异常batch。

额外按4个允许TRAIN序列的首/中/末身份抽12图，每图65,472候选；用数学等价的FP64解析逆重算SymKLD与分配代价，所选正样本12/12一致，零/负协方差行列式及伪零KL均0。固定两图KLD物理框梯度与该高精度版本最大差1.1322e−7；不支持把这次末态失败主要解释为矩阵求逆数值错误。12图所选正框score约.020～.029，均低于固定.05；有的靠近GT中心但尺寸仍错，不能用放宽score门槛恢复合格来源。

**目前能定位什么。** 已确认失稳发生在辅助检测器自身TRAIN优化，主分类曾大幅爆发；末态有明显的尺寸回归/共享特征梯度失衡，全局裁剪会同时压小分类与ATSS更新；现有保护只检有限性，未提前停止有限值的灾难性失稳。这些是可观察机制与工程缺口，尚未证明末态回归失衡先于分类尖峰，不能倒置因果或宣布固定系数就是唯一初因。没有证据把问题归给可靠性标签、容量不足或必须补数据。

**无法追溯的边界与下一步。** 实际辅助配置只在第24轮保存一次checkpoint，目录仅A24/latest；没有第8轮失稳前权重、optimizer/样本/RNG逐步快照，不能仅靠末态还原哪一更新先出问题。建议下一项只复现辅助A到首次异常窗口（固定TRAIN/原合同/seed），补逐步主分类正负项、主回归/共享特征及clip贡献、逐图身份/变换与分配、解码边界；保留异常前后权重及optimizer/RNG以便相同batch复放。这些快照只用于TRAIN因果诊断，不作选权或新OOF来源。先追踪原优化合同，不同时改LR、正样本数、损失权重或网络；复现/隔离后才提出一个辅助训练修复。不直接重跑A/B24轮或进入VAL/TEST。本轮未启动该复现。

**核验。** 诊断前后A24 SHA `d3b0da3655a6ab166e0a466ef1e5c4d542e24b1c17f89875c03e1bb27adc981c`一致，模型注册状态完全还原；诊断显式无optimizer。Mac独立复算384日志窗口聚合、分量loss/梯度范数平方、分类正负项、通道3×5聚合与源hash，全部一致；GPU梯度和预测仅由服务器计算，不声称Mac重放。首次通道统计夹具误按15通道连续分5组，已修为anchor×自由度并复算报告，没有将夹具问题归为训练错误。GPU reserved峰值3,186MiB，低于4GiB；不是长期显存平台证明。诊断结束GPU2/3各12MiB/0%，本次进程退出；服务器原工作区/暂存指纹b48cc0…/d54105…保持。没有修改模型源代码或启动训练。


### 44.12 辅助A失稳窗口复现与主检测同规则选权：本地实现（2026-10-11）

**授权与状态。** 用户因校园网暂不可达，要求先在本地修改两部分：复现辅助A至首次失稳窗口，以及辅助检测/midpoint选权。随后明确“权重选择不要这么严格，要和主检测差不多”；最终直接调用主检测既有`ckpt_sweep.select_best_checkpoint`及原`SELECTION_CONFIG`，撤销本轮最初拟定的额外95%/90%选权护栏和尺寸专用排序。本节仅记录实现与本地工程检查，没有连接服务器、启动GPU训练、选出新权重或确认新的根因。源码尚未提交/推送；另一几何窗口已有修改原样保留。

**入口与版本。** 新版本`port_reliability_oof_followup_v1`，入口`crane_project/tools/run_port_reliability_oof_followup_v1.py`及同目录`.sh`，合同`crane_project/utils/port_reliability_oof_followup_v1.py`，钩子`mmrotate/core/hooks/port_oof_trace_hook.py`，测试`tests/test_port_reliability_oof_followup_v1.py`。旧OOF入口/协议/87份源码SHA、固定第24/3轮历史结果、正式M/simple/policy不改。新工作流保存自己的源码及config SHA，不能以新选权重新解释旧失败。运行集中原`work_dirs/port_reliability_oof_v1/<RUN_ID>_<diagnose或selected>/`。

**第一部分：TRAIN失稳复现。** `diagnose`只从ImageNet新鲜初始化辅助A，保持seed1701、允许TRAIN1659帧、batch2/FP32/1024、两worker、随机等比例缩放/翻转、SGD/LR/warmup、原损失系数、分配日程及clip10。不加载失败A24或正式权重、不改优化方式。原24轮配置不变，新增钩子在7000次实际更新后停止，覆盖历史首个明显聚合异常6411–6460步附近；不自动延长。首次单步主分类loss≥5、主回归loss≥10、裁剪前全参数梯度≥1000或非有限时，在该步optimizer更新前隔离停止。这是由既有尖峰预定的诊断触发器，不是新的训练算法、最优系数或根因判据；首次触发不保证恰好是历史6411步，也可能未在此预算重现。

每步CPU日志记录实际图像、变换metadata、主分类/回归/ATSS loss、真实SymNFL正负项、正样本/分配/NFL计数、GT解码中心/尺寸/π周期轴向差、clamp前正框边界及编码残差、总/模块梯度、实际裁剪后范数及更新是否发生。头32步完整分量探针预热，之后每50步及6000步以后逐步分解分类/回归/ATSS梯度、共享特征方向余弦和编码dx/dy/dw/dh/da输出层范数；编码dh不称为规范短边。额外`autograd.grad`不写入参数`.grad`，实际更新仍由原严格optimizer hook负责；实际无扰动性及显存需在服务器核验。

只保留当前和前一更新的CPU状态，不积累GPU历史。异常保存`before_previous_update.pt`（若存在）、`before_anomaly.pt`、`after_forward_before_update.pt`；后者是前向后状态，**没有执行异常optimizer更新**。含权重、optimizer、Python/NumPy/Torch/CUDA RNG、CUDNN标志、未注册的assigner计数、训练模式及已经增强完成的真实batch；NFL buffer随权重保存。`replay`可无更新复放异常或前一batch，比较loss/梯度；不是恢复DataLoader预取及未来完整随机序列。快照不得作为选权候选、OOF来源或正式续训初始化。显存沿用32步平台预热、reserved零增长/allocated16MiB容差/4GiB上限，新增探针在optimizer前另检查；尚未测GPU开销，不能承诺与旧训练相同。

**第二部分：和主检测一致的离线VAL选权。** `selected`另立运行，检测器仍各24轮从ImageNet训练，每2轮保留一份，固定扫描epoch2/4/…/24全部12份，不平均、不只扫描存活的部分。选定一份检测器后，仅基于其允许TRAIN两尺度缓存训练自己的sigma1.5 midpoint3轮，保存并扫描head1/2/3，不做12×3联合搜索。检测器训练子进程不构造VAL/TEST；检测器完整训练结束并释放GPU模型/optimizer后扫描VAL。midpoint阶段先用冻结检测器缓存允许TRAIN、VAL及折外TRAIN特征，再仅以允许TRAIN更新head；VAL与折外TRAIN不进入优化。midpoint完成3轮并释放optimizer后才以VAL扫head权重，折外预测使用所选head；不读取TEST。诊断不自动连接完整A/B流程；先分析诊断，再决定修复与启动。

直接复用主检测/正式midpoint的`metric_protocol_v2`与原两阶段函数：跨域0.7Sim＋0.3Real，Weighted_R_center距候选最优不超过0.005、MCML≤5；可行域按`0.35*TDR+0.25*R_center+0.20*sim_ACI+0.20*(1-min(sim_A_RMSE,90)/90)`选择。无共同可行域时保留原两级fallback：先在MCML达标集合软评分，再在全体中优先最小MCML、较高R_center、较低Sim角度RMSE。并列沿用升序epoch的稳定顺序。**不新增尺寸正确率、95%覆盖或90%中心的选权硬约束。** 连续尺寸误差、双边正确全帧覆盖、输出覆盖及仅输出中心命中按域/逐视频记录，仅用于诊断。

“和midpoint类似”指完整VAL的离线几何/时序指标扫权重，不按训练loss或判断器FA挑辅助模型。底层评价器`mode='test'`仅启用全部时序指标；输入先核验全部原VAL身份，实际不读TEST。已暴露TEST只留以后冻结报告。完整VAL同时用于辅助选权与判断器开发评价，明确为共享开发VAL，不称独立验证。

**选权与来源可用性分开。** 选出辅助检测/midpoint后，其折外预测只来自该模型排除的原Real TRAIN序列；这些图像不进优化。GT仅监督/离线评价/生成判断器标签，不进在线forward。原来源工程检查、正确/错误两类支持、完整TRAIN拟合同3维simple对照、完整VAL同CR/同接受数FA及正确保留/连续FR评价继续保留；它们不是新增选权规则，选出权重不等于OOF材料合格。A来源不合格不训练B；正式M框、score、中心/方向标志及输出身份保护。辅助模型仅离线造TRAIN材料，部署新增辅助检测器数量为0，无自动TEST/部署。

**本地检查与限制。** 新18项测试14通过/4需Torch/MMCV跳过；原修复24项10通过/14跳过，原OOF20项19通过/1跳过，合计43通过/19跳过。纯测试执行主选权函数的实际AST提取定义，检查中心约束、评分、并列及原fallback，不另写替代选择器。合成OOF重放保存TRAIN/VAL只检查新版报告/policy身份和保护，不是新性能证据。新增运行时测试检查真实SymNFL值/梯度/调用计数、MMCV钩子先后顺序、异常前禁止参数更新、探针不污染backward及RNG/assigner恢复；服务器脚本先强制导入运行时再测试。Mac缺Torch/MMCV/OpenCV，实际MMCV prepare、完整离线指标、batch复放和显存未执行。4份Python语法、shell语法、git diff检查通过；旧87份源码SHA全部一致。未压缩本地源码、下载/覆盖权重或修改主要结果文件；不能写“失稳已修复”或“选权收益已成立”。

**网络恢复后的指令存档，当前未运行。** 指定项目/mmrotljj中先执行`bash crane_project/tools/run_port_reliability_oof_followup_v1.sh diagnose <新RUN_ID>`。脚本检查GPU2/3同时空闲，只用物理GPU3，显存异常即停。独立完整选权流程是同脚本`selected <另一RUN_ID>`，不与诊断自动连接。复放为`python crane_project/tools/run_port_reliability_oof_followup_v1.py --mode replay --out <诊断RUN目录/result> --gpu 0 --snapshot-boundary anomaly`（或`previous`）；运行前仍需空闲卡检查及CUDA_VISIBLE_DEVICES=3。未来诊断启动应沿用后台nohup/stdin=/dev/null，避免SSH断线终止。脚本退出自动将日志/JSON/逐帧记录压成一个文件到服务器work_dirs根目录，不含权重/快照/ROI缓存，不删除完整结果。

### 44.13 校园以太网DHCP恢复记录（2026-10-11 18:13–18:15，Asia/Taipei）

**范围与用户确认。** 本轮仅检查网络并记录，不修改网络设置、不提交推送、不启动或恢复服务器作业。用户明确确认本次恢复前没有操作，只是等待后恢复；因此不能归因于重启交换机、手动填IP或更换线缆。此前同学线路曾取得正常地址、随后也失败，不能仅据一次换线成功判定原网线损坏。

**已核实事实。** `configd`日志17:01:07曾实际设置`10.20.5.0/22`，17:01:19发布IPv4/DNS/DHCP；之后多次租约请求/SELECT超时并回落到169.254地址。18:08仍收到校园侧`10.20.4.1`的DHCP报文，但未建立租约；18:10:05记录`no server`。18:11:17收到有效ACK并进入BOUND，18:11:19实际设置`10.20.5.0`，18:12:03发布IPv4/DNS/DHCP。18:13现场读取租约确认：服务`Ethernet 2`/接口`en0`，掩码`255.255.252.0`，网关`10.20.4.1`，DHCP server identifier `10.11.0.41`，DNS选项`202.206.240.12/202.206.240.13`，网关ARP验证成功；这些是租约下发值，不等同所有应用实际采用的DNS。链路100baseTX全双工、MTU1500。`10.20.5.0`在此/22网段是合法主机地址，不能仅因末位0判为错误。

**恢复边界。** 服务器`10.20.5.146`的实际路由已从先前`utun4`恢复为`en0`，但18:13 TCP22的5秒探测仍超时；随后服务器ARP为`incomplete`，网关ARP有有效条目但两次ICMP无回复。故当前确认的是有线地址配置/校园直连路由恢复，不是SSH恢复、互联网全链路通过或服务器停机。ICMP无回复不能独自证明网关故障；服务器ARP未完成也不能区分服务器离线与端口隔离/链路问题。

**原因解释与待核实。** 直接失败环节是DHCP协商间歇性未完成；Mac自动重试后最终收到ACK，解释了没有手动操作也能恢复。校园DHCP/中继/交换机接入路径或本机网络处理的间歇性问题仍待核实，没有交换机端日志或受控对照，不能认定唯一根因。ClashX/代理此前仍启用、校园路由曾被TUN接管，是另一个访问因素，但没有证据将其认定为DHCP失败原因。先前只记录`no server`而推断完全无响应已由后续收包日志修正；不根据自动恢复要求手填历史动态地址。后续远程训练须先重新确认SSH实际可连接，不能把本记录作为服务器执行完成或训练启动证据。

### 44.14 非有限loss复放修复及VAL缓存表述校正（2026-10-11）

**授权与范围。** 用户要求修复上一轮审查发现的问题，明确服务器暂不可达、不运行远程任务。本轮仅本地修改新版followup合同、复放入口、相应测试及现有交接；没有连接服务器、提交推送、启动训练或读取新VAL/TEST。另一几何窗口已有改动保留。

**复放修复。** 新增实际复放入口使用的纯CPU比较函数`compare_replay_trace`：解析日志由`json_safe`保存的`nan`/`inf`/`-inf`字符串，有限loss保持原绝对容差1e−5＋相对容差1e−5，有限梯度范数保持绝对容差1e−4＋相对容差1e−5。NaN仅与NaN、Inf仅与同号Inf判为一致；有限/非有限或异号Inf不一致。非有限的算术差值为null，另记录实际值、原值、是否非有限及是否匹配，不用零替代异常，不改训练护栏或选权合同。复放`passed=true`仅代表重现保存的异常，原异常隔离状态与权重不可作为OOF来源的限制仍保留；不能解释为训练通过或失稳已修复。

**职责表述。** 原44.12“训练进程不构造VAL”已收窄到检测器训练子进程。midpoint实际先缓存冻结检测器的VAL/折外特征，然后只用允许TRAIN更新head，完成3轮并释放optimizer后才在VAL选head；说明与代码执行顺序一致，未为了文档改变算法或数据流。VAL选权依据几何/时序规则，可靠性VAL仍包含候选门限标定与效果评价，二者是共享开发VAL的不同用途，不称独立验证。

**本地验证与边界。** 新增5项复放比较测试全部通过，覆盖原有限容差、NaN/正负Inf的严格JSON往返及一致/不一致组合、有限与非有限双向不匹配、有限loss配非有限梯度、损坏文本拒绝。测试调用实际复放使用的比较函数，没有Torch/GPU模拟性能。3份受影响Python静态语法通过；旧87份源码SHA保持，原已通过的选权/训练检查未重复运行。真实GPU异常batch复放、MMCV钩子及显存未核验，待服务器恢复后执行；未产生新的训练或可靠性性能结论。源码仍未提交/推送。

### 44.15 辅助A有限失稳诊断：服务器后台启动（2026-10-11 21:14起，Asia/Taipei）

**授权与范围。** 用户确认服务器恢复，要求运行并确认“现在只是检测哪里出现问题”。本轮仅启动44.12的`diagnose`：辅助A新鲜ImageNet初始化，原TRAIN/seed/优化合同，最多7000次更新；不自动进入`selected`完整A/B、VAL选权、判断器训练或TEST。本次诊断快照不是可用OOF权重，启动正常不等于失稳已修复或可靠性改善。

**同步与后台证据。** 六份实现/测试/交接文件提交并推送`edd1d6a71e30bb975108c49b0afaf3ab3300ef7d`。服务器项目仍为`/media/omnisky/personal_files/ljj/symEOOD`。普通Git拉取被服务器既有127.0.0.1:17890代理拒绝连接，改用该提交的增量Git bundle传输/验证/快进，未修改代理配置；两端bundle SHA均为`fd081672159de38cf44641c3d230ea6bf9b9741d10d09136658f4d05618ad284`，同步后准确HEAD一致，临时bundle已删除。服务器已有worktree/index指纹仍为`b48cc0…`/`d54105…`，原暂存与未提交内容保留；另一几何窗口本地记录改动未纳入本次提交。

运行ID为`20261011_211408_A_instability_trace_v1_diagnose`，全部诊断产物在`work_dirs/port_reliability_oof_v1/`该运行目录内，外层launcher日志/JSON同置该父目录。后台wrapper PID2140249、训练PID2141926，实际cwd为上述项目，环境`mmrotljj`，训练`CUDA_VISIBLE_DEVICES=3`；stdin为`/dev/null`，训练SigIgn包含SIGHUP忽略位，独立session、输出重定向，普通SSH断线不会终止该已核实进程。不重复启动同ID、不根据此条静态记录判断后续仍存活。

**启动验证。** GPU2/3启动前均12MiB、0%、无compute进程，脚本测试后再次检查两卡空闲，仅使用物理GPU3。Torch1.13.1+cu117/MMCV1.7.0/GTX1080运行时导入通过；新版23项及原修复24项共47项服务器测试全部通过、无跳过。训练实际加载Real911＋Sim748，hook顺序为VERY_HIGH诊断→ABOVE_NORMAL严格optimizer→LOW更新确认。至本轮核验44步全部完成实际更新、无异常，损失/梯度有限、clip后范数约10；32步预热后的12步reserved保持2932MiB，更新结束allocated保持365.208MiB，GPU总占用瞬时约3252MiB，GPU2保持12MiB/0%。这是启动阶段平台证据，不是7000步全程稳定保证。原reserved零增长、allocated16MiB容差、4GiB绝对上限与异常更新前隔离均继续生效。

**当前未完成。** 诊断仍运行，尚无`completion.json`/`exit_code.txt`，未捕获失稳、未复放异常batch、未作唯一根因结论。7000步不重现也只记预算内未重现，不称训练问题解决。结束后应先核对completion、触发步是否执行optimizer、各分量梯度/分配/GT解码和快照SHA，再决定是否需要无更新单batch复放及一项明确修复；不自动重跑完整训练。脚本退出将日志/JSON压成一个文件置服务器work_dirs根目录，不含权重/PT快照/data，完整产物保留；本轮尚无完成压缩包可回传或清理。查看日志：`tail -n 50 -F work_dirs/port_reliability_oof_v1/20261011_211408_A_instability_trace_v1_diagnose/run.log`；逐步证据为该目录`result/A/trace/updates.jsonl`。

**离开前复核与同步边界。** 后续读取107步均已更新，预热后75步reserved仍全为2932MiB，step107主分类0.3133、主回归1.1530、ATSS0.0856、裁剪前梯度131.869、裁剪后约10、无触发，训练PID仍在运行。运行记录提交`25a5a70`已推送；期间另一几何窗口提交`860c6da`进入同一main，额外记录bundle命令因可能夹带非本轮提交被自动审批拦截，未传输。服务器保留`edd1d6a`诊断实现，不声称两端最新HEAD同步；该边界不影响已启动且源码固定的诊断。后续不要为了同步文档改动运行中的诊断源文件。

### 44.16 辅助A第6452步梯度异常捕获及无更新复放（2026-10-12）

**完成与范围。** 用户要求读取服务器已结束的诊断。服务器同一项目内读取完成状态，训练于10月12日01:04:10（Asia/Taipei）隔离，实际训练从前日21:15:30起约3小时49分；没有达到7000步或完整24轮。6452条逐步记录中前6451步真实更新，第6452步在optimizer前停止，状态`ANOMALY_QUARANTINED_BEFORE_OPTIMIZER`。训练子进程exit1来自预定TraceStopped，外层诊断识别预期停止后exit0；不称训练/来源性能通过。A/B完整训练、midpoint、判断器、VAL选权/可靠性评价与TEST均未启动，正式M/policy不变。

**单步事实。** 首个触发理由仅为裁剪前总梯度1179.705≥预定1000，主分类loss0.55956、主回归8.00122、ATSS0.23586，均有限、其他两个loss阈值未达到。该步两图为`real_seq06_00342`/`real_seq13_00116`。三个loss分别求全模型梯度，范数为分类58.02198、回归1175.35022、ATSS6.72665；总梯度的backbone917.45886、neck427.83334，说明回归的影响已传入共享特征。分量范数不能直接加和或当作可相加百分比。编码回归输出weight的dh通道平方范数占87.197%，仅为该输出层编码通道统计，不能称全模型贡献或规范短边贡献；前一6451步da通道占54.69%，不支持单通道唯一根因。

异常前最后局部稳定段6411–6445的总梯度中位12.093，最大58.171；6446步回归梯度95.728、总92.857，随后总梯度6448/6449/6450/6451为133.398/250.993/402.312/582.660。以上已执行更新仍裁剪到约10，6452未执行任何更新。当前证据显示触发阶段由回归梯度主导，而不是主分类loss先达到触发阈值；旧训练在随后50步聚合窗口分类爆发，不据此宣布本次捕获的回归异常是最初且唯一原因。

**可排除的具体环节与限制。** 全部逐步loss/梯度有限；32步预热后到异常的reserved始终2932MiB，无显存增长/OOM。编码GT解码中心/尺寸最大误差3.052e−5px、π周期轴角最大误差1.192e−7rad，最后100步正框clamp前越界计数0；异常每图仍1个正样本，LR0.0025/NFL tau1，未在该步发生O2M→O2O或LR切换。没有观察到编码—解码错配或当前正框边界裁剪原因；这些检查不能证明所有标签语义或分配选择完全正确。全局clip10只缩放合成梯度，不重新平衡分类/回归/ATSS方向；loss值有界及有限检查也不等于梯度变化受控。

**复放与独立核验。** GPU2/3空闲后，在GPU3依次调用既有`--mode replay --snapshot-boundary previous/anomaly`，恢复保存的6451/6452真实增强batch、权重、optimizer/RNG及未注册分配计数，但optimizer_updates=0。两次报告passed，loss/总梯度差值均精确0、原快照SHA保持，没有训练或参数修复。服务器原始日志包SHA`39f2f44315dc46d9596ce6b1c1450e3c07f7422e47d5c18a286563d6a9879c81`已安全回传解包（28项、无PT/PTH），两份后来产生的复放报告单独读入同一运行结果目录，无权重回传。Mac核验6452步连续性/更新边界、分域图像身份、有限性、GT误差、显存和两份复放标量；与旧完整训练124个50步窗口比较，分类/回归/总loss及梯度差均≤5.01e−6，符合原五位小数舍入。不能将Mac标量复算称GPU复放。

本地审查明细在原运行目录`local_diagnostic_review_20261012.json`，两份复放报告在`result/A/trace/replay_previous/report.json`及`replay_anomaly/report.json`。服务器完成前后原worktree/index指纹一致，复放结束GPU2/3均12MiB/0%。仅本次已核验日志压缩包按精确路径/SHA清理，完整结果/逐步记录/三份快照保留；不删除其他窗口或历史材料。本节本地记录，未改算法、未提交/推送或同步其他窗口提交。

**下一步建议，尚未执行。** 不立即重训24轮或另换判断器。复用已捕获6451/6452快照，进一步追踪主回归loss→实际正框几何→编码残差→共享特征的梯度，检查尺寸/角度敏感性与实际所选正样本；区分解码/几何敏感、分配变化和既有更新累积后，再提出一项辅助训练修复。该阶段可用保存的真实batch做无更新归因，不需要再次从头复现几小时、增加新数据或修改正式M。尚未确认唯一根因，也没有新的可靠性或深度性能结论。


### 44.17 两份TRAIN快照梯度链归因与主回归预算候选：已测，未采用（2026-10-12）

**范围。** 用户授权复用6451/6452快照，定位“回归损失→预测框→编码→共享特征”的放大环节后做一项修复并连接服务器测试。本轮只在允许TRAIN的两个保存批次/四个增强视图上操作，无DataLoader重采样、VAL/TEST图像加载或评分、完整A/B/midpoint/判断器训练。GT仅用于这些离线监督/归因，不修改正式M/simple/policy或线上API。源快照SHA沿用44.16，所有新结果集中`work_dirs/port_reliability_oof_v1/20261012_gradient_chain_v1/`；这是诊断材料，不新增数据划分或OOF权重。

**实现与核验。** 新增`run_port_reliability_oof_gradient_chain_v1.py`、`port_reliability_oof_gradient_chain_v1.py`、`test_port_reliability_oof_regression_budget_v1.py`、`PortOOFRegressionBudgetHook`及测试。真实前向仍仅一次分配/NFL调用；捕获实际正框、原归约/权重、预测框物理梯度、解码Jacobian及VJP、编码梯度、FPN/回归输出输入与五物理坐标的参数梯度向量。归因无参数更新，两次原loss/总梯度均重现，五坐标向量相加的相对误差6451/6452为9.403e−7/7.542e−7；不是把范数相加。源码提交依次`a4f4c15`/`564959e`/`a27120f`/`df52f98`/`8f31439`，均已推送main；服务器通过只含本轮文件的增量Git分支同步，测试源码末态对应main`8f31439`、服务器测试提交`6b81199`，文件SHA独立核对相同。未同步其他窗口的几何/文档整理改动。

**具体归因事实。** 6452步两个正框规范长边179.907/247.913px，对应GT38.722/48.910px，即4.646/5.069倍；短边20.757/26.580px，对应25.814/23.274px。KLD原始shape项23.384/11.993、center_p＋center_q项2.728/9.993，压缩前距离26.112/21.986。物理长边导数0.02507/0.01077，经指数解码的Jacobian179.907/247.913，传到编码dh的导数4.50977/2.67088；宽高交换后该dh在这两帧实际对应规范长边，不能称短边通道。长边物理坐标的参数梯度向量范数1037.881，角度648.706；两者可抵消，不能相加归因。6451角度向量范数831.393也大于该步总回归586.680，进一步说明不能锁定单一通道。主回归直接通过`retina_reg(FPN)`影响共享特征，6452的回归backbone/neck范数912.253/427.066。

FP32物理梯度与主轴解析FP64参考的相对差仅2.744e−7/9.195e−7，协方差det相对差≤3.885e−7；解码VJP与实际编码反向一致。因此这两个快照不支持逆矩阵精度错误，观察到的是过大长边/角度偏差、合法的解码尺度导数和共享参数回传共同放大。尚不能证明这些因素是整个原训练最初且唯一根因，也未排除分配、历史momentum或后续批次作用。

**唯一候选修复。** 主回归分量先用`a=min(1,10/||g_reg||)`预算，再和未缩放的其他loss梯度相加，随后仍做原全局clip10及SGD/momentum。10继承已有clip范数，不选新系数或搜索权重；前向loss/标签/解码不变，只有辅助训练反向改变。钩子仅显式导入/实例化在本次有限测试，未改原完整OOF入口/正式配置。最初`total+(a−1)*reg`在FP32的1e8梯度夹具中期望13却得到14.9209，已修为先反向`total−reg`，再逐参数加`a*g_reg`，最终夹具精确13，避免近−1系数抵消。旧实现结果留在finite64，不按其更低末态loss择优；最终候选固定取数值正确版finite64_fix1。

**有限对照合同与工程修复。** 两臂均从before_anomaly的同一权重、optimizer/momentum和RNG开始，交替使用已保存anomaly/previous两批次，不模拟原未来DataLoader序列，不重置momentum。最初16步夹具错误将预热缩为2步，控制第4步新增2MiB reserved而退出，实际4次更新，原报告保留在finite，不能当模型/候选失败结论。已恢复原32步预热、reserved零增长/allocated16MiB容差/4GiB上限，并固定两臂64步，增加32步平台观测；没有放宽护栏。控制只用原有限值检查＋全局clip，允许记录后续有限尖峰，这是隔离的反事实，不是恢复原诊断运行或取消其停止护栏。候选最终实现变更后只重跑候选，未重复未变的64步控制；实际step计数来自optimizer.step调用，避免旧控制后更新显存检查失败被误记未更新。

| 固定64步/末态，仅四个TRAIN正样本框 | 原全局clip控制 | 主回归预算，最终数值实现 |
| --- | ---: | ---: |
| 首步裁剪前梯度 | 1179.705 | 60.465 |
| 全程最大裁剪前梯度 | 5,937,262 | 124,782.906 |
| 超过原1000诊断触发器步数 | 25 | 2（第8/20步） |
| 两批末态平均总loss | 1.14815 | 0.58569 |
| 两批末态平均回归loss | 0.50513 | 0.17078 |
| 正样本长边/短边相对MAE | 16.676%/32.584% | 10.013%/7.662% |
| 正样本双边≤10% | 0/4 | 1/4 |
| 正样本中心MAE/轴角MAE | 11.196px/33.507° | 4.338px/7.359° |
| GPU reserved峰值 | 2758MiB | 2878MiB |

以上正样本框是TRAIN分配得到的监督对象，**不是最终交付检测输出、完整TRAIN/VAL性能或可靠性FA/FR**。两臂均完成64步，预热后reserved分别保持2758/2878MiB，结束GPU2/3均12MiB/0%。源快照不变，临时模型权重全部丢弃、不保存为可用OOF来源。原控制重复两批也能恢复，说明单个尖峰不必然造成不可逆崩溃；真实视频样本切换/更新累积仍需另证。

**未通过与决策。** 候选主回归贡献确实≤10，但第8步分类loss505.322、回归0.60456/预算前回归梯度118.367，总梯度124782.906；受预算的回归最多10，说明非主回归部分仍可产生巨大梯度，不能仅靠压主回归解决全部失稳。未单独分解该步分类/ATSS梯度，不将总梯度数值直接标为分类梯度。原1000/分类5停止合同下候选仍会隔离，因此未采用为完整辅助训练修复，不启动完整A/B、不自动加第二修复或换结构。仅认定局部梯度/四视图拟合有改善信号；完整TRAIN稳定、OOF来源合格、VAL可靠性与深度收益均未证明。

**工程与回传。** 最终10项服务器测试全部通过，Mac缺Torch：2项CPU通过、8项运行时跳过（不称Mac梯度复验）。服务器包SHA`b7c629478a37a23fb69f7e7f1f49f1e522d672d27a79d88eac90921c5fab9284`安全回传到同一运行目录，22项、无权重/PT；Mac核对5份测试源码SHA、6份JSON结果SHA、两次归因重现/向量重建、64步计数、同初始化/批次顺序、末态聚合与25/2尖峰计数一致。明细为analysis_summary.json/local_review.json及result、finite64、finite64_fix1子目录。原工作区/暂存b48cc0…/d54105…保持，其他窗口正在整理的本地文档原样保留。本次包核验后清理，两端完整结果及原快照保留；没有更新正式主要结果成绩或论文稿。

**同步隔离。** 本次源码已推送origin/main（代码末态8f31439）。因另一个窗口正在整理文档，仅将本轮源码/可靠性记录同步并推送`codex/oof-gradient-chain-v1`；服务器实际工作目录已切换该任务分支，main引用保留任务开始时25a5a70，避免主分支快进冲突或夹带几何文档。此调整只处理本轮创建的提交引用，原worktree/index内容不改；任务分支可正常追踪origin同名分支。当前任务记录在该分支归档，本机主工作区文档整理改动与本节同时保留，未整包提交其他窗口修改。

### 44.18 现有回归预算修复的健康初始化预防验证：第90步资源护栏停止，机制未验收（2026-10-12）

**授权与范围。** 用户授权验证现有修复能否预防失稳，修改代码并按原流程Git同步/服务器运行。只运行辅助A，原ImageNet初始化、seed1701、原1659张允许TRAIN、batch2/FP32/原增强、SGD及24epoch学习率日程保持；从第一步采用44.17相同主回归预算10及原全局clip10，最多7000次实际更新，首次异常停止。不继承6452权重或SGD动量，不启动B/midpoint/判断器、不加载VAL/TEST图像或选权、不修改正式M/simple/policy。原TRAIN/VAL角色元数据与来源SHA审查仍复用，不能称未读取任何VAL元数据。

**增量与合同。** 新增`run_port_reliability_oof_prevention_v1.py/.sh`、`port_reliability_oof_prevention_v1.py`、`PortOOFPreventionHook`和11项测试，独立入口不接入旧完整A/B训练。反向仍先计算真实主回归梯度，其他loss正常反向，显式加预算后的回归梯度。原始总梯度由其他梯度＋未缩放回归梯度计算并记录，停止判断使用预算后、全局clip之前的实际混合梯度≥1000；保持分类loss≥5、主回归loss≥10、任何非有限值以及原32步预热/显存护栏。护栏均在本次optimizer.step之前。原24epoch配置仅保持学习率/NFL日程，实际7000次即终止，不由MMCV完整24epoch ETA推断本轮时长。

不再每步完整CPU模型快照或额外原总loss反向/密集分量归因：仅保存小规模可变buffer/RNG/assigner运行状态。异常时参数和momentum尚未更新，将当前参数与前向前buffer合成准确的前向前快照，另存前向后快照及实际增强批次。正常每步记录标量JSON；不保存可作为OOF来源的训练权重。通过仅表示该A窗口稳定，不代表完整TRAIN稳定、OOF来源合格或可靠性FA/FR收益；失败则本次不自动增加修复或预算。

**检查与同步。** 本地3项不依赖Torch的合同检查通过，8项Torch检查跳过；服务器11项新测试通过，另10项原回归预算/解析参考测试通过。测试大梯度夹具最初独立重建回归表达式，与实际捕获共享张量不一致；已修正为复用捕获回归张量，不改变训练修复。实际hook验证loss停止时零更新、前向前buffer和assigner还原、正常预算＋clip更新，以及显存失败时参数/momentum均零更新。源码main提交b3b3af0/ff2a7f4/949f42d已推送；服务器隔离任务分支`codex/oof-gradient-chain-v1`同步797b29f/b46d0a8/e835f53，避免夹带另一窗口文档整理。原87份来源身份继续核对。

**运行。** 服务器实际目录`/media/omnisky/personal_files/ljj/symEOOD`、环境`mmrotljj`，后台PID513734，SIGHUP忽略、stdin=/dev/null、新session；SSH断线不结束后台进程。初查第三张GPU有其他200MiB任务，未启动或停止该任务；再次确认物理GPU2/3均12MiB、0%、无计算进程后启动，训练只使用物理GPU3。脚本prepare后再次检查GPU2/3。结果统一`work_dirs/port_reliability_oof_v1/20261012_A_budget10_prevention_v1/`；script退出时自动生成`work_dirs/port_reliability_oof_prevention_v1_20261012_A_budget10.tar.gz`，无PT/PTH/数据软链。完成后需读取状态、回传复算并记录，再仅删除本次压缩包；完整结果和异常快照保留。本轮随后在第90步因资源护栏停止，不能判定失稳预防效果，也没有可靠性性能结论。

日志（服务器项目目录内）：
```bash
tail -n 50 -F work_dirs/port_reliability_oof_v1/20261012_A_budget10_prevention_v1/run.log
```
prepare阶段尚无run.log时读取同目录launcher.log。初始时长估计3～5小时。启动后独立核验五份源码SHA两端一致，原A与本轮15项配置/资源比较均一致；已观察到82次实际更新，32步预热后50步reserved保持2878MiB，最后主分类/回归loss为0.25557/1.40677，尚未触发异常。前50步日志time=1.834s/step，按7000步估计约3.6小时，暂按3.5～4小时安排；MMCV显示10:07:16是原24epoch全程ETA，不是本轮预算。该早期状态不代表经过6452原失稳窗口或最终修复通过。

**最终状态与独立核验。** `UPDATE_OR_RESOURCE_FAILURE_STOP`，实际完成89次更新，第90步在optimizer.step之前退出；没有重新运行或放宽显存容差。该步两张允许TRAIN为sim_seq08_00484/00650，主分类loss0.343703、主回归0.698738、总loss1.078433，原总梯度137.655、预算后混合28.363、全局clip后10；90步均无loss/非有限/混合梯度1000触发。reserved由预热平台2878MiB增加为2880MiB（2MiB），而allocated389.722MiB低于预热最大392.597MiB。因此是严格缓存平台合同未通过，不是已观察到模型loss/梯度失稳、OOM或已证明持续泄漏；未达到原6452附近窗口，不能宣布修复有效/无效。更新成功的第33—89步共57步reserved均2878MiB。原始/混合梯度峰值425.875/197.571，90次主回归预算均启用；这也不是完整训练或检测交付质量证据。

服务器仅CPU读取前向前/后快照核验183个参数张量和SGD optimizer/momentum完全相同、快照SHA不变，异常记录与快照trace一致；两快照保持diagnostic_only，不作为OOF来源。进程PID513734已退出，当时GPU2/3回到12MiB、0%。wrapper退出码0表示按已登记边界停止并成功归档，**不表示PREVENTION_WINDOW_PASS**；真实结论须读取completion.status。原正式M/policy及A/B历史身份保持。

结果包补入CPU分析后SHA为`83f0cc13959b44786b3cbb60687aa7dc4001b7224da9bc20850387afd0eb2e96`（初始包哈希另保留initial_archive.sha256）。安全回传同一运行目录，31个tar成员、无PT/PTH/数据软链；Mac独立核对10份源码SHA、15项原A配置比较、90行次序与89次更新、57步缓存平台、loss/梯度未触发、异常资源信息、工作区和暂存指纹不变。记录analysis_summary.json/local_review.json；没有把CPU检查写成Mac Torch复验。完整结果与服务器PT异常快照保留，包核验后按用户授权清理。运行记录只更新本节，不改正式主要结果或论文稿。

**下一步边界。** 用户要求缓存预热后不增长，本次已按要求停止，不自动续训、不改变容差或预热长度。若后续继续，应先解决/确认资源检查可执行性，再重复同一预防合同；不能将这次资源退出当作回归预算的机制失败而叠加第二种训练修复。仍不启动完整A/B、VAL/TEST或可靠性判断器训练。

### 44.19 第90步缓存定位、标量有限值检查及同合同重启：1731步资源退出，预防效果未验收（2026-10-12）

**授权。** 用户要求先定位2MiB增长，若无大问题重新运行既有修复的健康初始化预防验证。本轮只处理辅助TRAIN诊断检查的缓存开销，不添加第二种训练修复、不改原32步预热/零reserved增长/allocated16MiB容差/4GiB上限、loss与梯度停止阈值、预算10、7000步、TRAIN/seed1701/模型/SGD/LR日程，仍不启动B或VAL/TEST。

**零更新定位事实。** 新增`diagnose_port_oof_prevention_memory_v1.py`，读取44.18第90步前向前快照（SHA bd0ab7…），保留其optimizer状态以匹配存储，但不执行optimizer.step。固定原版/标量版各40次同一实际增强TRAIN批次，每次恢复前向前buffer、RNG与assigner运行状态；参数逐张量核验不变，源快照SHA保持。Python行级缓存追踪显示原检查中两次`torch.isfinite(g).all()`各新增2MiB GPU缓存，原40次末态reserved均2782MiB；首次之后不持续增长。该追踪只定位保存批次的新分配，不重现原90步全部allocator历史，因此不能宣布两次分配就是原晚发2MiB的唯一根因，也不能凭一次增长证明泄漏。

**实现增量。** `finite_gradient_norm`使用L2标量范数有限性检查，NaN/Inf或范数溢出仍停止；主回归预算/其他梯度/显式逐参数组合/全局clip10/实际SGD更新均不改。本次独立`PortOOFPreventionHook.clip_grads`保持原MMCV裁剪操作及日志，有限性预检采用同一标量检查，不修改通用PortStrictOptimizerHook或旧正式配置。版本标记`allocation_revision=scalar_norm_finite_checks_v1`单独保存并校验，以保留原运行身份。未清空GPU缓存或增大预热/容差。main提交a617f7a/6f5262e，隔离服务器任务分支fd5feb9/51463bb，均已推送；保留另一窗口已有工作区/暂存改动。

**验证事实。** 服务器13项测试通过，新增标量检查NaN/Inf/缺梯度拦截及64维梯度与原实现逐位相同、所有isfinite检查仅处理标量的测试；原实际hook零更新停止/前向前buffer/显存失败参数与SGD动量不变/正常裁剪更新继续通过。Mac3项合同检查通过、10项Torch跳过。标量版40次GPU零更新重放的总loss、原始梯度、混合梯度、回归梯度、缩放与预算范数每次均与原版JSON浮点值完全相等（loss1.078433、raw137.655365、mixed28.362972）。reserved均2780MiB，比原版少2MiB；allocated范围两者相同387.722～388.222MiB，没有持续累积。仍不能保证真实不同批次的完整7000步缓存不增加，原护栏持续执行。定位材料统一在44.18原结果目录下`memory_diagnosis`/`memory_diagnosis_scalar`及`memory_diagnosis_comparison.json`；不是新增数据角色或模型候选性能。

**首次重启尝试未开始训练。** `20261012_A_budget10_scalar_prevention_v1`后台launcher PID603596启动时物理GPU2出现其他窗口进程598655（几何TEST），物理GPU3空闲。脚本严格按GPU2/3都空闲要求在测试/prepare/训练前退出，exit_code3，不能写成训练失败或新的预防结果；不停止该进程、不读其TEST结果、不占用其显卡。该尝试日志/归档保留原身份，待空闲后另立运行ID避免覆盖。

**用户更新GPU授权与实际重启。** 用户随后明确“继续就行，不用在意第三张卡，显存占用小”，因此仅第四张物理GPU3为空闲检查对象；不再以GPU2任务阻断，不改变训练显存护栏。脚本变更main df598af／服务器任务分支63540e1已推送同步。重新使用新ID`20261012_A_budget10_scalar_gpu3_prevention_v1`，后台PID613570，SIGHUP忽略、stdin=/dev/null、新session，mmrotljj环境/指定服务器目录不变。服务器13项新版测试＋10项原梯度链/预算测试通过后执行prepare和来源检查，仅A从ImageNet重启，不续用89步权重或momentum、不启动B。新的源码SHA/配置/pipeline/运行身份另存；旧失败运行不改结论，原第90步异常快照继续保留。完成后需读completion.status，回传复算、更新本节并仅删除本次结果压缩包；在此之前不能宣布失稳预防有效或可靠性性能改善。

当前日志：
```bash
tail -n 50 -F work_dirs/port_reliability_oof_v1/20261012_A_budget10_scalar_gpu3_prevention_v1/run.log
```
prepare期间可看同目录launcher.log。仍预计7000步约3～4小时，实际速度和首次异常可改变完成时间，原24epoch日志ETA不是本轮预算。

**重启早期观测（非最终通过）。** 服务器重启后15项配置/资源比较全部一致，10份源码SHA核对通过。已观测119次实际更新，第33—119步共87步reserved保持2878MiB，尚无completion；前89步样本、loss、原始/混合/回归梯度、缩放及裁剪后梯度与44.18旧轨迹完全一致。第90步的同批次loss/梯度也与旧记录一致，现在optimizer.step成功且reserved仍2878MiB，说明这项检查分配优化已越过先前资源退出点；不能外推所有7000步或原6452失稳窗口。明细`configuration_comparison.json`/`early_observation.json`，诊断包SHA63156661…本地80条重放记录独立核对相同；GPU占用退出包SHA06cc01a9…核对exit3、无result/无训练。两类已完成分析包按用户授权两端清理，完整目录/日志/allocator JSON/旧PT快照保留；新后台训练未结束，最终结果压缩包仍待脚本退出后生成。

**本轮最终结果（2026-10-12，用户要求读取服务器核验）。** `20261012_A_budget10_scalar_gpu3_prevention_v1`实际完成1730次更新，第1731次（epoch3）在optimizer.step之前触发`UPDATE_OR_RESOURCE_FAILURE_STOP`，不是7000步完成。运行从10:20:06到11:11:44约51分38秒，原后台PID613570已退出，GPU3释放。第33—1730步共1698次成功更新reserved均2878MiB；第1731步reserved3152019456bytes＝3006MiB，比预热平台增加128MiB，严格零增长护栏停止。该步检查时allocated390.222MiB低于预热allocated峰值392.597MiB；没有逐阶段peak_allocated记录，因此不能排除临时工作空间峰值，也不能认定128MiB增长就是正常缓存或持续泄漏。与44.18的2MiB检查张量事件不同，不套用其唯一根因解释。

第1731步允许TRAIN为real_seq12_00125/sim_seq08_00502，主分类loss0.104432、主回归0.126782、总loss0.235034，原始/混合/裁剪后梯度均4.175124，回归梯度4.053685≤10、scale1；并非loss或梯度尖峰触发。全1731次尝试主分类loss最大0.535065、主回归最大3.374839、原总梯度最大425.875184、混合最大197.571152，均有限且没有分类5/回归10/混合梯度1000触发；1093次尝试实际启用主回归预算。loss下降与早期平稳只是TRAIN运行事实，不是检测来源合格、判断器FA/FR改善或修复已预防原6452失稳。

**结论与边界。** 原失稳窗口6452未达到，7000步预防合同没有通过；也不能据此判断主回归预算修复无效，因为退出原因是资源护栏。正式M/simple/policy、旧A/B与其他窗口模型身份保持，无B/midpoint/判断器训练或VAL/TEST评价，不自动重跑、不放宽显存护栏、不增加第二种训练修复。若继续，应先定位这次128MiB的临时峰值/缓存扩展与持久引用，再决定如何让同合同运行到原窗口；不能仅凭阶段末allocated较低直接宣布无问题并续训。

**终态证据与回传。** 服务器仅CPU读取before_anomaly/after_forward_before_update快照，SHA与completion一致，iteration均1730；183个参数张量及SGD optimizer/momentum完全相同，after.trace与第1731行一致，证明该步未更新。wrapper exit0只表示预设边界成功隔离/归档，并非PREVENTION_WINDOW_PASS。现有23项服务器测试状态与10份源码SHA均核对，不重复运行未改测试。CPU分析保存analysis_summary.json；诊断快照不作为OOF来源。原生成包SHA9ea6418d…另存initial_archive.sha256，纳入本轮分析后包SHA`398f31abe12b49acbcc18fac7baf9834c9581fc2aead8cfe6c79c49cf4d4e011`，安全回传同一运行目录32个成员，无PT/PTH/数据软链。Mac独立复算1731行连续次序/1730更新/1698步平台/1093次缩放/最大loss梯度/停止资源信息，核对15项配置与10份源码SHA、工作区/暂存指纹不变。local_review.json明确没有Mac Torch复验；没有以exit0或测试通过替代预防通过。完整结果与服务器异常PT保留，包核验后仅清理本次两端压缩包；本轮只更新可靠性运行记录，不改论文主要结果或算法。

### 44.20 用户允许暂时显存增加后的资源判定修复与预防验证：第6135步分类loss触发停止（2026-10-12）

**授权更新。** 用户明确“查看诊断结果，修复这个问题，要是暂时的显存增加可以接受，继续后续的验证现有修复能否预防失稳”。本轮据此撤销本次入口的reserved预热后零增长判断，允许4GiB上限内缓存扩展；不再把每次cached增长等同泄漏。不修改原通用MemoryPlateau/PortStrictOptimizerHook、旧实验终态、正式M/simple/policy、模型/损失/主回归预算10/全局clip10/SGD/seed1701/TRAIN身份/24epoch日程/7000步，不新增第二种训练修复，不启动B/midpoint/判断器或VAL/TEST。

**阶段诊断事实。** 上轮中断前新增`diagnose_port_oof_prevention_memory_v1.py --expected-step1731 --all-phases`，明确校验保存快照iteration1730、SHA与原completion。固定40次GPU零更新，恢复保存的实际增强批次/buffer/RNG/assigner，不做optimizer.step，参数与快照SHA不变。逐行及backbone/neck/main head/aux head/forward/预算反向/clip记录显示该批次首次分配后reserved均2782MiB，检查点allocated387.722～388.222MiB，最高阶段活跃峰值2607.930MiB；冷进程没有重现原1730步allocator历史中的晚发128MiB，所以仍不锁定其唯一原因、不凭重放稳定证明整个训练无泄漏。诊断的GPU同步和Python逐行追踪会增加开销，仅诊断入口启用，不在新训练中运行。源快照只诊断，不作为OOF来源。工具main fbf8df9／服务器994d4b2已推送。

**资源规则独立版本。** `memory_revision=cache_aware_resource_v2`，只接入PortOOFPreventionHook，配置与prevention.json/completion均显式记录，旧配置不能被新入口重新解释。继承32步预热、检查点活跃显存16MiB容差及4GiB绝对上限；cache扩展只记录。固定64步活跃allocation窗口，每32步评估前/后32个值的上中位数；后中位数高于预热活跃峰值＋16MiB，且比前中位数增加≥4MiB，连续两次评估满足时停止。上中位数避免单次跳到稳定较高平台被连续两次误判；短暂活跃尖峰可接受，但任何观测的current/peak allocated/reserved>4GiB立即停止。趋势分辨率以下的慢/小增长仍受硬上限约束，不声称数学保证永无泄漏。窗口仅CPU整数deque(maxlen64)，没有GPU张量历史；不清空cache、不人为预占内存、不更改反向。每步额外保存allocator current/peak/inactive_split/active/retry/OOM标量，即使guard抛错也保留原始资源证据。loss5/10、混合梯度1000、非有限值及更新前隔离继续保持。

**验证。** 本地8项CPU合同/资源测试通过、10项Torch跳过；服务器18项全部通过，覆盖NaN/Inf、梯度逐位等价、FP32大梯度、实际hook资源失败时参数/SGD动量零更新、缓存扩展与短暂尖峰、32种单次稳定跳变对齐、持续活跃上升、current及peak4GiB上限、错误配置/计数。不重复未变原测试，后台复现脚本自带另10项原预算/解析参考检查。新guard零更新重放已保存90行/1731行检查点资源记录，分别接受2/128MiB cache扩展，均无持续活跃增长触发；旧运行status不改，旧逐阶段peak历史不可得，不能声称其历史全部峰值也通过新cap。重放辅助命令最初误用系统python缺numpy，已切回既定mmrotljj，不安装包或改环境。结果`20261012_A_budget10_scalar_gpu3_prevention_v1/cache_aware_guard_record_replay.json`。源码main1aa434d／任务分支57afd5b已推送，两端运行源文件同步。

**新运行。** 物理GPU3启动前12MiB/0%、无计算任务；按用户允许GPU2共存授权只检查GPU3，不停止其他任务。服务器目录与mmrotljj环境不变，后台PID1609876、SIGHUP忽略、stdin=/dev/null、新session，SSH断线不结束。新ID`20261012_A_budget10_cacheaware_gpu3_prevention_v1`；仍从原ImageNet/seed1701健康初始化，非从1730/6452快照续训，避免未保存的DataLoader未来序列/采样状态改变原有限合同。最多7000实际更新，首次训练异常或新资源规则触发停止，无自动完整A/B续跑。旧快照/目录/规则保留。当前仅已启动，不能宣布预防成功或可靠性收益；需到达原6452窗口并完成7000后另验。

日志：
```bash
tail -n 50 -F work_dirs/port_reliability_oof_v1/20261012_A_budget10_cacheaware_gpu3_prevention_v1/run.log
```
脚本退出后单个结果包位于work_dirs根目录，分析回传核验后仅删除该包，完整结果保留。约3.5小时预算，实际进度/首次异常决定耗时，MMCV24epoch ETA不是本轮预算。

**启动后独立核对。** 新配置除resource guard外的14项模型/优化器/LR/seed/增强/采样/工作流/初始化/全局clip比较均一致，10份源码SHA检查通过；原资源规则和新规则显式同时写入configuration_comparison.json。已观察225次真实更新，225步的样本、loss、原始/混合/回归梯度、缩放及裁剪后梯度与44.19保存轨迹完全一致，预热后reserved仍2878MiB；记录的活跃峰值2617.475MiB、缓存峰值2878MiB，未触发新保护。early_observation.json明确原6452窗口尚未到达、最终预防结果待定；这个阶段不能代表7000步通过或可靠性提升。

阶段诊断与资源记录重放包SHA`0b55a5725b3cb6d444772d988cb1a6855305ea3397de274d6f1cad65105e88fb`安全回传到原1731运行目录（无权重/数据），本地独立核验40次零更新/峰值/缓存及90＋1731＝1821行guard重放输出逐项一致；旧峰值历史不可得仍明确披露。cacheguard_analysis_local_review.json保存复核。本轮18项server测试＋后台脚本10项原预算测试通过，Mac8项CPU通过、10项Torch跳过；不将Mac统计核对写成Torch梯度复验。已完成分析包在核验后仅删除两端包文件，保留所有JSON/allocator分段与旧快照；新7000步验证仍在运行，未生成或清理其最终结果包。


### 44.21 第6135步分类梯度归因与三任务预算代码修复：实现及有限检查，未完成健康初始化预防验证（2026-10-12）

**上轮终态。** `20261012_A_budget10_cacheaware_gpu3_prevention_v1`完成6134次更新，第6135步主分类loss12.672535超过原上限5，在该步反向与optimizer.step前隔离；主回归6.193066、总loss18.946337。状态`ANOMALY_QUARANTINED_BEFORE_OPTIMIZER`，不是7000步通过，也未到原6452步。约17:42停止，进程退出后GPU3恢复12MiB。cached仅1731步增加128MiB后保持3006MiB，没有OOM或持续active增长；本次停止不归因于显存。原始快照SHA为before `f16c750d8f093160ae600ffce7da516c68865e41922995b90b74a4b318ee2d3c`、after `373cfbee88ada5f06c43e584c69db7e85c9f95f6f19bd2e8bc9c2722c7a4735e`，结果/快照保留，不作为OOF来源。

**事实与推断。** 新增`diagnose_port_oof_classification_stop_v1.py`，复用保存的真实增强TRAIN批次、模型/SGD/RNG/assigner状态，不加载数据集或更新参数，不下载预训练。三类前向损失与第6135行精确一致。主分类梯度6570.454、主回归3048.284、辅助27.604；分类loss来自P3，正样本0.254473、负样本12.417924，约98%来自负样本；两图各1个正样本，avg_factor2，197个负anchor概率≥0.5，logit有限。编码GT误差≤7.63e-6、周期角误差0，主分类对reg输出层梯度0、对backbone/neck梯度4890.584/4106.286。可确认原预算只限制reg，未限制分类贡献；不能据此认定正样本数、某视频、SymNFL公式或分类分支是唯一初始根因。归因峰值reserved2846MiB，参数及源快照未变。

**代码修复，尚未采用。** 独立`port_oof_task_budget_v1.py`及`PortOOFTaskBudgetHook`，三组为主分类、主回归、原辅助损失总和。各自g_t=grad(L_t)，alpha_t=min(1,10/||g_t||)，显式累加sum(alpha_t*g_t)，然后保留原global clip10及SGD。10沿用原预算尺度，未做系数搜索。前向损失/标签/分配/模型/增强/seed不改，原loss5/10、非有限值及资源护栏不放宽，旧hook/旧配置及正式M/simple/policy保持。缩放贡献先各自物化再相加，修复测试发现的mul与融合add舍入不一致；不使用total减大项的FP32取消。独立`run_port_oof_task_budget_prevention_v1.py`显式校验新revision、源码及配置SHA，不允许旧配置被重新解释。新目录准备过程中独占write冲突已修复；旧准备失败目录保留，新`fresh_entry_fix1`prepare/check通过。

**必要验证。** 本地语法和diff检查通过；Mac无Torch，7项本地测试中1项通过、6项跳过，不称本地梯度验证。用户更改运行要求前，服务器7项测试全通过，覆盖小梯度等价、大梯度三项约束、反向取消、零/未用参数、NaN/Inf、实际hook保持loss/显存失败前参数与momentum零更新。异常快照零更新两臂原前向完全相同，旧仅reg预算混合梯度6588.457，新三项预算23.341782；两臂仍有原main_cls警报，不能称异常快照通过生产护栏。

**固定64步反事实，不是生产续训。** 两臂从同一已失稳快照及SGD动量、同一保存增强批次开始，重复64次，不代表真实未来采样序列；只诊断局部机制，不保存可用权重。有限反事实显式记录而不执行原loss警报，非有限/显存保护仍生效；生产hook的loss警报保持执行。旧仅reg预算：64步、11步原警报，末步cls0.327831/reg3.496814/total3.841447；新三项预算：64步、9步原警报，末步cls0.276469/reg2.114996/total2.416561。两臂均有限，不能以末步下降或警报稍少证明健康初始化稳定、检测质量或可靠性收益。结果统一在44.20目录的`classification_stop_diagnosis`和`task_budget_checks`下；未新增数据角色，未读取VAL/TEST。

**最新用户要求优先。** 用户随后明确“先修改代码吧，不需要去服务器上运行”。此要求到达时，新健康初始化7000步有限验证刚启动，ID`20261012_A_task_budget_gpu3_prevention_v1`、supervisor2516628；已立即仅终止经核实的本任务进程组。最终确认完成5次更新、该组无剩余进程、GPU3为12MiB，`user_stop.json`标记USER_REQUESTED_STOP；shell exit0不能解释为预防通过。之后不再运行服务器实验。新7000步、完整24轮A/B、midpoint、判断器、VAL/TEST均未完成。仍需另获运行指令后从健康初始化验证，本轮只保留本地修改与已有事实，不宣布失稳已彻底解决。

源码main `cc6282a`/`7a91557`/`e2d85ba`/`6270e04`在用户变更运行要求前已推送；服务器隔离任务分支源码为`1aef83e`/`d3da69e`，既有worktree/index二进制指纹核对保持。服务器GitHub代理不可用时仅用Git bundle传源码，不修改其网络配置。失败/反事实材料仍保留服务器原目录；未回传新权重、未清理未核验结果包、未修改论文结果或新建本地回传副本。

### 44.22 三任务预算修复获重新授权，健康初始化有限训练启动（2026-10-12）

**授权与改进核对。** 用户明确本次检查已完成，要求查看针对性改进并启动服务器训练，撤销44.21末尾的暂不运行要求。已核对独立三任务预算入口、反向实现和hook：主分类／主回归／原辅助损失总和各限梯度范数10，再按原global clip10及SGD更新；前向标签、模型、增强、loss5／10异常拦截及cache-aware资源规则保持。本地main `6270e04`与服务器隔离分支`d3da69e`的入口／预算函数／hook／测试四份源码SHA完全一致，复用本版已通过的7项服务器测试，不重复完整诊断。上一轮仅回归预算6135步分类异常及5步USER_REQUESTED_STOP身份均保留。

**本次有限范围。** 仍按已实现合同，仅辅助A、健康ImageNet初始化、seed1701、原1659张允许TRAIN、batch2／FP32，保留原24epoch学习率日程但最多7000次实际更新；首次loss／非有限／资源异常停止。不继承6134步或旧异常快照及动量，不自动转入完整A／B、midpoint、判断器、VAL／TEST或选权。当前正式M与simple/policy保持。本轮是训练稳定性预防检验，不是可靠性性能评价。

**后台启动身份。** 指定目录`/media/omnisky/personal_files/ljj/symEOOD`、mmrotljj环境、物理GPU3（第四张卡）；启动前12MiB／0%、无计算进程，保留用户允许GPU2共存授权。新运行`20261012_192025_A_task_budget_gpu3_prevention_v1`，supervisor PID2737538、独立session、SIGHUP忽略、stdin=/dev/null、stdout/stderr重定向；先fresh prepare／来源check，再次检查GPU3空闲后训练。没有覆盖上次停止目录。运行脚本／launcher.json／配置及来源SHA集中在本次目录，退出时自动生成一个结果包至work_dirs根目录（不含权重／数据）；读终态、回传核验后才清理该包。

日志入口：
```bash
tail -n 50 -F work_dirs/port_reliability_oof_v1/20261012_192025_A_task_budget_gpu3_prevention_v1/run.log
```
prepare阶段看同目录`prepare.log`／`launcher.log`。尚未完成7000步，不宣布失稳已解决或可靠性FA／FR改善。服务器既有工作区／暂存差异指纹启动前分别为`b48cc0f9…`／`d54105b5…`，退出脚本继续核对，不改动这些既有差异。

**新运行早期核验。** fresh prepare／check通过，prevention.json的14份源码SHA逐一匹配，三项预算各10、允许TRAIN1659张及fresh_ImageNet_only身份正确。已观察70次实际更新，32步预热后的38次reserved保持3178MiB，最后检查点allocated约451.40MiB，记录峰值allocated2862.18MiB／reserved3178MiB；没有loss异常、OOM或completion。显卡总占用包含CUDA上下文，与Torch allocated／reserved分开解释。仅早期观测，不保证后续显存平台或7000步通过；当前early_observation.json保存在服务器本运行目录。

### 44.23 用户停止三任务预算训练后的等价显存精简：零更新检查通过，未重启长程训练（2026-10-12）

**停止事实。** 用户要求停止当前后台训练并简化，但不能改变三任务失稳修复。已核实并仅停止`20261012_192025_A_task_budget_gpu3_prevention_v1`的进程组2737538；训练进程2741503及其workers退出，实际完成290步，GPU3恢复12MiB，`user_stop_memory_simplification.json`标记人为停止而非失稳或预防通过。旧记录保留，未自动续训。第33—75步旧三任务版active446.484～453.154MiB、reserved3178MiB、峰值2863.430MiB；前一仅回归版同阶段active388.972～392.472MiB、reserved2878MiB、峰值2616.850MiB。两版方法不同，不把这组早期显存比较作为算法性能对照或泄漏定论。

**精简范围与不变项。** 方法`task_budget_revision=per_task_gradient_budget_v1`不变，新增独立`task_memory_revision=release_loss_graphs_optional_raw_v2`，新配置和完成报告显式校验；旧来源不能被新入口重新解释。三组损失及顺序、各自预算10、显式FP32缩放累计、global clip10、SGD/LR/seed/模型/增强/TRAIN角色、原loss5/10、非有限任务及交付梯度、缓存/活跃趋势/4GiB保护保持。训练默认不分配完整未缩放梯度累计缓冲；各任务raw norm/scale仍保留，完整raw总范数只在诊断通过`measure_raw=True`获取。未测量时raw总范数及其峰值/超限计数为null，不填假零、不将各项norm相加冒充实际norm。反向结束后detach runner总loss，释放hook的loss/reg引用及局部terms/aux/total，保留batch、buffer/RNG和actual p.grad供更新前隔离。不调用empty_cache、不扩大容差/上限，不修改在线API。

**必要检查。** 本地语法及diff检查通过；无Torch，9项单测中1项CPU合同通过、8项跳过。服务器9项全部通过，包括measure_raw开/关时交付梯度、global clip后的SGD参数及momentum逐位一致，loss引用释放不改变梯度/scalar/batch/buffer；实际hook的loss/资源失败仍在optimizer前停止，旧NaN/Inf/大梯度/反向取消检查继续通过。新prepare/check通过，18项配置、修复参数及TRAIN身份/增强对照一致，未启动train模式。

**实际模型零更新对照。** `check_port_oof_task_memory_v2.py`复用同一已核验第6135步TRAIN快照、SGD状态、实际增强批次、buffer/RNG/assigner。旧算法从Git6270e04读取冻结源码，SHA与被停止run的记录一致。两臂独立GPU3进程，各4次复放，未做optimizer.step、未加载VAL/TEST。4次原前向loss、三项norm/scale、mixed norm、全参数裁剪前后梯度SHA全部逐位相同；模型参数及源快照未变。旧峰值allocated2849.883301MiB，新2729.032227MiB，降低120.851074MiB；reserved3044→2924MiB，降低120MiB；末次active444.093750→387.459473MiB，降低56.634277MiB。无OOM。结果仅证明该固定批次的等价及显存改善，不保证未来所有批次不涨，不等于检测或可靠性收益。

材料统一在被停止run的`task_memory_v2_check`下；两份report及comparison小型JSON回传原对应目录，本地独立复算4次哈希一致和显存差值，`local_review.json`标明无Mac Torch重放。未回传权重、未生成本轮结果压缩包。代码main931b98d、服务器隔离任务分支7c4fd7d；服务器其他暂存内容指纹一致。先前被停止run及其归档保留，没有恢复290步权重或再次后台启动训练。

**时间与后续边界。** 等价显存检查只需短程零更新，不需要3—4小时。判断健康初始化能否预防原失稳，仍需另一次固定7000步TRAIN窗口，耗时数小时，应按精简后的实际训练速度估计；本轮未启动，不能用零更新通过代替。完整24轮A/B、midpoint、判断器及VAL/TEST也未启动。正式M/simple/policy及论文主要结果不变。

### 44.24 精简版重新获授权并启动，最终第6067步分类loss越界停止（2026-10-12—13）

**最新授权与范围。** 用户确认精简不改变失稳修复，并授权验证其作用，撤销44.23末尾的本轮不重启要求。三任务预算版本仍为`per_task_gradient_budget_v1`，各10、global clip10、SGD及原loss/非有限/资源停止规则保持；仅使用`release_loss_graphs_optional_raw_v2`显存实现。复用已通过的9项服务器测试、4次同批次零更新哈希等价及18项配置对照，不重复未变检查。仅辅助A、健康ImageNet初始化、seed1701、1659张允许TRAIN、batch2/FP32；原24epoch LR日程保留，但最多7000次实际更新。不是290步或6134步快照续训，不启动完整A/B、midpoint、判断器或VAL/TEST，不改正式M/simple/policy。

**实际后台启动。** 指定服务器目录和mmrotljj环境不变，物理GPU3启动前12MiB/0%、无计算任务，允许GPU2共存，不操作其他任务。运行ID`20261012_200133_A_task_memory_v2_gpu3_prevention_v1`，supervisor2914811、独立session、SIGHUP忽略、stdin=/dev/null、日志重定向；fresh prepare与check后再次检查GPU3空闲才训练。新目录不覆盖旧停止记录。到7000步或首次loss/非有限/资源异常自动停止，不自动转入完整训练。退出自动归档到`work_dirs/port_reliability_oof_task_memory_v2_20261012_200133_A_task_memory_v2_gpu3_prevention_v1.tar.gz`，无PT/PTH/数据；最终分析核验前不清理此包。运行时不修改服务器源码/Git工作区，保护既有工作区和暂存指纹。

**早期实际更新等价核对，非稳定性通过。** 已观察64次更新，新运行前64步的图像顺序、losses、三项原梯度norm/scale/预算结果、mixed norm、裁剪后norm与44.22旧三任务run逐项完全一致；14份源码SHA全匹配。32步预热后已观察32次，reserved均2952MiB、最后active389.959473MiB、历史峰值allocated2739.079102MiB，cache增长事件0、OOM0，没有completion或异常停止。显卡总占用约3272MiB，低于旧三任务版本早期3498MiB；不能保证未来缓存不再变化。`early_observation.json`集中在新运行目录。前64步一致补充支持实现等价，未到6135/6452失稳窗口，不能宣布7000步通过、长程稳定或可靠性收益。

日志：
```bash
tail -n 50 -F work_dirs/port_reliability_oof_v1/20261012_200133_A_task_memory_v2_gpu3_prevention_v1/run.log
```
日志ETA仍按完整24epoch，不能当作本轮7000步剩余时间。首50步均值约2.475秒/步，含启动和预热开销；第51—100步均值2.510秒/步，20:06:38的100步日志正常，reserved2952MiB、GPU总占用3272MiB。按此速度剩余6900步约4小时49分，正常预计10月13日00:55左右，后续速度或首次异常可改变耗时。GPU进程正在后台运行，SSH断开不会结束已验证的任务。

### 44.25 精简版三任务预算的完整有限验证核验：未通过，梯度与显存约束生效（2026-10-13）

**终态事实。** `20261012_200133_A_task_memory_v2_gpu3_prevention_v1`完成6066次真实更新，第6067步（epoch8）在原main_cls≥5保护触发后、反向与optimizer.step之前隔离；状态`ANOMALY_QUARANTINED_BEFORE_OPTIMIZER`，未达到7000步，也没有越过上一轮6135或原6452停止位置。completion写入时间10月13日00:16:17；trainer_exit1是受控停止，外层exit0只表示识别并归档成功，不是训练通过。09:08服务器核对GPU3已恢复12MiB、无本任务GPU进程。本次不改算法或保护阈值、不自动重新训练，不进入B/midpoint/判断器/VAL/TEST。

**触发批次与解释边界。** 第6067步两图为`sim_seq08_00415`和`sim_seq08_00332`；main_cls5.234626、main_reg1.036298、auxiliary0.029738、total6.300662，全部有限。该步在loss保护阶段停止，所以没有其原始/交付梯度记录；不能凭前一步梯度说该步没有梯度放大，也不能宣布已定位相同根因。此前成功更新main_cls最大2.725291，最后6066步main_cls0.239519，第6067步的loss突变可能包含批次差异；目前只确认预先规定的loss门槛未通过，没有证明持续崩溃、NaN/Inf或哪个视频是唯一原因。

**修复作用分开判断。** 6066次已执行更新各任务预算结果均≤10（浮点容差），混合梯度最大19.525143，global clip后最大10.000000954，均有限；未裁预算的任务梯度最大main_cls832.001831/main_reg9494.941406/auxiliary198.138306，说明不能把raw高值消失当成成功指标。三项预算正确限制了交付梯度，但没有保证未来分类loss始终小于5。显存精简的实际收益保持：预热后active388.959473～392.959473MiB，历史peak allocated2740.033691MiB，reserved最大2954MiB，仅1015步cache增加2MiB，无持续active增长或OOM。问题不是缓存失控或预算未执行；整体失稳预防合同仍失败，不采用为合格OOF来源。也不能据此将梯度预算或整个可靠性研究宣布为根本无效。

**独立核验与来源。** 6067行次序连续、前6066行optimizer_updated=true、最后false，completed计数一致；14份运行源码SHA全部仍匹配，服务器worktree/index前后指纹一致，没有热换代码或错误版本。仅CPU加载异常前后快照并构建同结构模型取参数名称，183个参数张量、SGD参数组及momentum逐位相同，iteration均6066，after.trace与末行完全一致。before SHA`6a91766904ac5fed5fa9ba70782943bcbc2f270696c48036ad0a22e3f023af16`、after SHA`ae9bc6ec1b9684bb3bef37f9ff81195bfed9c1cbddeeb210536713ef202118fb`与completion匹配。13项完整性核对通过，不能将它写成训练预防通过；没有GPU重放或新训练。

**材料与记录。** 服务器`analysis_summary.json`集中在原run目录。原28成员无PT/PTH/数据的结果包SHA`590161d675b48c14d6f0839d5868316fc3ef65b42397782185b2f7d0e4de9be5`安全回传到相同run目录，Mac独立复算行数/更新边界/有限性/最大mixed norm/状态，`local_review.json`明确未执行本地Torch。完整日志、配置、JSON及服务器PT快照保留；核验和记录后只删除本次两端tar.gz，不删除旧实验、权重或结果目录。未将失败候选写入论文主要结果。

**下一步建议，尚未实施。** 不继续加一层梯度预算、不放宽main_cls上限或重跑完整训练。先复用第6067步固定快照做一次零更新分类正负损失、logit及正样本分配核对，确认越界是否由少数难例/大量负anchor还是持续输出退化主导，再据证据确定有限修复。当前结论应写成“交付梯度与显存受控，但预定loss稳定性检验未通过”，不声称可靠性FA/FR已有改善。

### 44.26 第6067步零更新分类归因与相反动量重置候选（2026-10-13）

**授权与范围。** 用户授权复用44.25快照，做一次不更新参数的分类损失归因并修复。本轮只在原服务器项目/mmrotljj、空闲物理GPU3进行保存批次重放、必要单元测试与配置prepare；没有optimizer更新真实模型，没有启动新的7000步或完整A/B训练，没有读取VAL/TEST，也未改正式M/simple/policy、检测几何或深度公式。服务器旧工作区及暂存内容保持原二进制指纹。

**零更新归因事实。** `diagnose_port_oof_classification_stop_v1.py`显式支持旧仅reg配置与当前三任务配置，恢复同一before快照、实际增强batch、SGD/RNG/buffer/assigner状态。main_cls5.234626/main_reg1.036298/auxiliary0.029738/total6.300662与停止行精确一致，参数及SGD状态、源快照SHA保持。分类正项0.0620606，负项5.1725661，占98.8144%；几乎全部来自P5层，两图各1个正样本、avg_factor2，GT编码还原最大误差1.91e−6px、周期角误差0。所有负样本概率均<0.5；P5两图各3071个负样本，前100个负样本合计损失约0.87257，仅占该层负项约16.87%，不是几个极高分背景独占。空间乘子覆盖的负项合计约0.18666，仅为该层负项约3.61%，没有证据把主要放大归给空间KLD乘子。

主分类原梯度1333.0621，backbone651.2752、neck1070.3414、分类输出455.2630，对主回归输出梯度0；主回归178.5981、辅助0.65578。共享cls/reg方向余弦−0.36834。只确认本批次的背景累积、共享特征敏感与任务冲突，不宣布正样本数、场景或分配公式是唯一根因，也不将raw梯度大小等同实际交付更新。

**更具体的更新机制。** 只读还原三任务预算、global clip及普通SGD（含weight decay与历史momentum），没有写参数或动量。当前裁剪梯度＋weight decay与β×旧momentum的全模型内积约−48.1973；SGD旧momentum范数22.3603，假设更新的参数改变量范数约0.050534。分类loss的一阶方向贡献为当前分类−28.92536、回归＋9.64149、辅助−0.40177、历史动量＋29.25101、weight decay−0.000160，总＋9.56520；若移除旧动量，总为−19.68581。这里是局部导数及假设更新方向，不能把它加到5.234626当作实际下一步loss，不能证明动量造成了先前所有异常。明确的工程限制是：梯度预算只约束本次`.grad`，SGD携带的历史动量仍可能与当前更新方向相反。

**单项可选修复，尚未采用为合格OOF来源。** 新revision`opposing_sgd_momentum_restart_v1`，只在辅助独立入口启用`PortOOFMomentumRestartSGD`和对应hook。沿用原三任务各10预算及global clip10；原main_cls≥5/main_reg≥10、非有限、混合梯度1000、4GiB/cache-aware护栏均保持。通过原护栏之后、实际SGD.step之前，计算A=Σ〈g_clip＋wd×θ, β×旧momentum〉；A<0时仅对本步有梯度的参数原地清零旧momentum，然后执行原SGD。A≥0或无历史momentum时完全沿用普通SGD；未参与本步更新的参数及其动量不动。学习率0.0025、momentum0.9、wd0.0001、日程、模型、标签、分配与数据合同不改，没有额外网络、整模型梯度缓存或新loss归一化。这个判据保护的是当前预算后方向，不能保证原始多任务目标或分类loss的有限步下降；重置过频也可能影响收敛，后续须报告次数及检测质量。

**实现及验证。** 服务器9项测试全部通过，覆盖动量同向时与普通SGD逐位等价、反向重置、零/无动量、不参与更新参数、非有限/不支持SGD配置、配置合同，以及真实hook正常更新记账和loss/资源停止前不修改参数或动量。本地仅1项纯合同通过、8项Torch检查跳过，不声称Mac执行Torch。`check_port_oof_momentum_restart_v1.py`在实际模型上重新计算原三任务梯度和global clip，只读判定A约−48.19702，与独立CPU方向复算差<1e−3；原loss精确保持，参数/SGD/source snapshot均不变，optimizer_updates=0、restart_applied=false。原main_cls警报仍存在；这不构成训练通过或低loss收益。peak allocated2717.5713MiB、reserved2920MiB，仅证明本次零更新资源检查，未验证长期平台。

**配置与来源。** 独立`run_port_oof_momentum_restart_prevention_v1.py`仅prepare及内置check通过，允许TRAIN1659帧、seed1701、fresh ImageNet、原24epoch LR日程、最大7000真实更新，不继承异常权重/optimizer，不自动转入B/midpoint/判断器/VAL/TEST。最终配置在原run的`momentum_restart_prepare_final/result`，没有训练产物。原run及早期归因/prepare版本保留，不覆盖；最终依据为`classification_attribution_v4/report.json`、`momentum_restart_final_check/report.json`与`local_review.json`。本地复算正负loss、占比、更新方向符号、18项来源pin/配置SHA及六份本次源码SHA全部一致，未执行Mac模型重放。没有生成或下载本轮压缩包，直接回传小型JSON、日志及配置到对应目录，原PT不回传。

本地main源码提交`7a4e6fd`已推送；服务器保持既有隔离分支`codex/oof-gradient-chain-v1`提交`c35df33`，六份源码SHA相同，不能声称两端HEAD相同。服务器既有worktree/index指纹仍为b48cc0…/d54105…；用户原交接草稿未夹带进源码提交。本节作为执行记录，不写入论文主要结果或性能表。

**后续边界。** 分类归因和有限修复实现完成，但健康初始化7000步预防验证未启动，失稳预防、完整OOF来源合格和可靠性性能收益均未证明。后续应在独立新run从健康初始化验证该单项增量，不用第6067步已越界快照续训宣称“预防成功”，不放宽原loss门槛、不追加另一预算或同时换数据/模型。

### 44.27 两臂各32步短程恢复对照：动量重置没有整体分类优势，单次loss越界不等于持续失稳（2026-10-13）

**授权与解释修正。** 用户授权采用短程诊断，并指出单次loss5.235未必是不稳定。这个区分成立：44.25仍保留“原固定7000步/单次loss警报合同未完成”的历史事实，但不能把它直接改写为已证明持续崩溃；44.26零更新梯度方向也只提供局部机制线索。短程通过最多支持保存状态附近的恢复增量，不等于长期预防、完整辅助来源合格或可靠性FA/FR提升。本次未改变生产警报5/10等合同。

**固定对照。** 新独立`test_port_oof_momentum_restart_short_v1.py`，控制为原三任务预算＋原SGD，候选唯一增量是相反动量重置。两臂在两个独立GPU3进程，从相同before快照、SGD/RNG/buffer/assigner计数开始，重复同一保存的两张增强TRAIN图各32次真实诊断更新，随后各做一次无更新前向测量最终loss。不存在新数据角色、原未来采样复现或OOF权重。用户明确授权独立副本的有限更新；单次有限loss5/10警报逐行记录而不在这个反事实入口停止，非有限/混合梯度1000/4GiB资源仍硬停，原production hook保持。这些诊断副本不保存权重、不续原run、不读VAL/TEST、不进入正式判断器训练。

运行前固定局部增量条件：两臂完整且有限；候选更新一次后的分类loss更低；后32次前向分类均值、最大值、最终分类及最终回归均不比控制差。不临时挑一步、追加步数或改重置条件。工程检查及33行完整性与增量条件分别判断。

| 同一TRAIN批次测量 | 控制（三任务预算） | 相反动量重置 |
|---|---:|---:|
| 相同初始main_cls |5.234626|5.234626|
| 一次更新后main_cls |30.324038|0.410431|
| 32次更新后的main_cls |0.442916|0.456384|
| 初始之后32次前向main_cls均值 |1.587264|2.071000|
| 初始之后main_cls最大值 |30.324038（第2前向）|52.899914（第6前向）|
| 32次更新后的main_reg |2.056349|0.149636|
| 单次loss警报前向数（含相同初始） |2|2|
| 实际动量重置次数 |0|23/32|
| 实际运行时间 |79.13秒|84.01秒|
| peak reserved |2926MiB|2924MiB|

**事实与新发现。** 两臂各32次更新完成，33条前向记录次序正确、最后一条不更新，全部loss/梯度有限，无OOM；任务贡献均≤10，混合梯度最大15.7893/19.6193。候选确实消除了原首次30.324尖峰，但第6前向转而在P3出现52.900尖峰（负项52.57577、约99.39%）；当时P3的98,304个负样本平均概率约0.10595、45,913个≥0.1，全部<0.5。控制首次尖峰主要在P5，候选后续尖峰主要在P3。因此，只约束历史动量与预算后混合梯度的内积，不能保证所有层的背景响应与分类loss受控；23次重置说明本批次优化历史改变频繁，不能凭这一计数证明它是根因。

两臂都在反事实中自行回到约0.44～0.46分类loss，说明5.235警报及更大的峰值在这个重复批次下可恢复，不足以单凭峰值宣布持续失稳。反过来，重复批次loss回落也不能证明不同TRAIN批次上的稳定、正框得分/几何及输出覆盖正常，不能推翻旧失败A24实际低分/尺寸错误等功能退化证据。需要把“有限数值异常”“可恢复尖峰”“持续功能退化”分开记账。

**结论。** 候选5项固定局部增量条件只通过首次分类改善及最终回归保护，其余3项不通过；不采用为解决分类loss波动的整体修复，不启动44.26准备好的7000步运行，也不继续调重置系数/预算。原三任务预算身份保持，其长期预防及OOF来源资格仍未证明。后续优先明确真实训练的异常/恢复判据：单次loss越界属于预警证据，持续异常和检测功能退化需另有预先规定的观测与隔离规则，配合健康快照；本轮只是建议，没有修改生产规则或训练流程。

**复核与材料。** 本地/服务器各4项纯逻辑测试通过；Mac独立复算两臂66条记录、任务预算、每层正负loss重构、5项固定比较及源SHA，`local_review.json`明确无Mac Torch执行。材料全部集中在原run的`momentum_restart_short32_v1_20261013_01/{control,restart}`与comparison/log下；原PT SHA保持、权重未回传、未生成压缩包。两臂启动前GPU3空闲，后台nohup/setsid运行，总实际进程时间约163秒；exit0表示完成诊断，不是增量通过。结束GPU3恢复12MiB，服务器原worktree/index逐位相同。源码main`d2f450d`已推送，服务器保留隔离分支`95f28d0`，源文件一致，原交接草稿未夹带进源码提交。未写论文主要结果，未新增长期训练。

### 44.28 当前6066步快照实际检测功能：允许TRAIN只读推理严重退化（2026-10-13）

**授权与范围。** 用户授权检查当前6066步快照实际检测功能，并询问是否还需3—4小时。新增独立`check_port_oof_snapshot_detection_v1.py`，只恢复`20261012_200133_A_task_memory_v2_gpu3_prevention_v1/result/A/prevention/before_anomaly.pt`完整model状态，不重新加载ImageNet、不构造optimizer、不反向、不更新参数、不加载midpoint。完整1659张允许TRAIN仅标准尺度推理，原score>.05、max_per_img=1保持；排除折、VAL/TEST均未构造推理数据集。GT仅离线计算指标。不是来源晋级或正式权重选取。

**工程经过。** 首次入口复用旧预处理校验器时，它只接受`PortStrictOptimizerHook`，因此在推理前退出；日志完整保留。实际配置先由`check_task_config`核验，在只供预处理校验的副本中恢复旧hook名称及其memory配置，不构造任何优化器、原cfg/model状态不改。修正版在独立`detection_function_6066_v1_20261013_r2`目录执行，物理GPU3启动前空闲，后台运行。全部1659帧实际推理159.62秒，exit0；首次接口错误不计为模型训练失败。新增脚本未Git提交/推送，服务器既有tracked工作区与index二进制diff前后完全相同。

| 允许TRAIN分组 | 输出/全部帧 | 中心<15px/输出 | 双边均≤10%/全部帧 | 长边相对MAE | 短边相对MAE |
|---|---:|---:|---:|---:|---:|
| Real |911/911|1/911|18/911|34.48%|43.92%|
| Sim |748/748|0/748|0/748|30.03%|57.46%|
| 总体 |1659/1659|1/1659|18/1659|32.48%|50.03%|

每组输出覆盖均100%，因此全帧中心正确覆盖数与表内中心命中数相同。总体输出中心正确率及全帧中心正确覆盖0.0603%，双边正确覆盖1.0850%。逐视频seq06：输出466/466、中心1、双边正确18；seq12：141/141、中心0、双边正确0；seq13：304/304、中心0、双边正确0；seq08：748/748、中心0、双边正确0。连续尺寸误差仅按输出统计，没有将漏检标成尺寸错误。总体中心误差中位436.09px、P95 759.86px，score中位0.33423。Real长/短边有符号bias +25.59%/+35.19%；Sim −12.68%/+51.63%。

**独立核验。** 参数及所有注册buffer推理前后哈希一致，PT SHA仍为`6a91766904ac5fed5fa9ba70782943bcbc2f270696c48036ad0a22e3f023af16`；optimizer更新0。全部1659张GT与原TRAIN证据最大差1.79e-7，角色/完整身份一致。按身份固定四序列各首/中/末共12图，以标准`model(return_loss=False,rescale=True)`重放，预测与主报告最大差0.000488px，手工按实际sx/sy还原与`rescale=True`的差0。推理peak allocated343.38MiB/reserved412MiB，进程结束GPU3恢复12MiB。Mac仅CPU独立复算1659条预测、全部分组中心/双边计数与连续误差，未模型GPU重放。小型JSON/日志直接回传，没有PT或压缩包。

**结论与停止边界。** 6066步状态在允许TRAIN的标准推理功能严重退化，不是仅靠单次loss5.235判断“崩溃”；低/可恢复的训练loss也没有建立正确交付框。不能确定首次退化发生在哪一步，不能推断所有早期权重已坏或某损失是唯一原因。该快照不作合格OOF来源，不启动新7000步、完整A/B、midpoint、判断器或VAL/TEST，不从该状态续训。建议暂停继续堆叠优化器修复，收束当前辅助来源路线；不把本项辅助模型失败归给正式M或将其写成可靠性整体无效。正式M/simple的身份与已有成绩保持；此次没有新增FA/FR性能结果。

全部结果集中原run的`detection_function_6066_v1_20261013_r2`，主要文件为`result/report.json`、`result/predictions.jsonl`、`interface_audit.json`、`local_review.json`、`interpretation.md`及日志。原首次失败目录及历史快照均保留，未写入论文主要结果。

## 文档整理：可靠性主报告合并（2026-10-12）

用户授权合并可合并的四份可靠性文档。旧`OBB观测可靠性与连续输出.md`与`base_v3_v51_focused_paper_complete_pipeline_20260915.md`的重复机制、流程、σ1／epoch23完整TEST、SHA与命令合入[可靠性统一报告](obb/可靠性实验主要结果与对比分析_20261006.md#history-sigma1)，相同表只保留一份；普通B、σ1、σ1.5各自policy与结果保持分开。文献参考保留独立，执行交接不合并。

项目内相关入口更新，DINO历史说明指向其真实的替换前归档；原归档及数值、原始产物、代码、权重、policy、阈值保持。文档合并没有新训练／推理／拟合／TEST读取，不改变第30—44节或其它版本的采用／停止／未完成状态，不将尚未确认的候选收益写入当前成绩；实际大小论文稿未改。


## 45. 因果时序尺寸证据有限检验及废弃OOF文件精确清理（2026-10-13）

### 45.1 运行前固定合同

用户明确结束辅助来源训练，授权本地代码修改、Git同步、原服务器目录测试及删除辅助路线无用文件。第44节各原失败、未完成和来源资格保持；不恢复辅助A/B、midpoint、判断器训练，不加载第6066步快照。正式M仍B24＋sigma1.5/epoch03及匹配simple。

只读取既有完整TRAIN2558及VAL887预测与原policy，来源e804241c…/38d914f…绑定；没有新的检测推理、Torch训练或TEST预测。主证据预定为当前规范长/短边log尺寸相对过去3个相邻观测log尺寸中位数的最大绝对差；一步变化仅记录，不选优。GT只离线生成双边>10%错误标签及拟合工作点。历史API只收当前框、观测索引、原图大小及显式重置，无GT/未来/天气/域路由。width-height规范及等比例单位不变性检查通过。无框、观测号断开、原图分辨率改变、视频切换均重置；seq12采集拼接122由sampling_manifest绑定。数据是抽样观测索引，不称原视频相邻帧或秒数。

单一TRAIN工作点：只用历史充分且原simple已经接受的正确TRAIN输出，各总体/域/视频求95%全同分接受门限，取最大值保护各组；不调窗口、不换特征、不用VAL选门限。离线temporal veto仅将原simple已接受且变化超限的size标志作拒绝诊断；历史不足保留原标志，中心/方向/框/score/数量均原样，不部署新policy。完整报告保留全部帧与MISSING，另列历史充分的共同集合排序，不能借暖启动删除难帧。报告原policy、veto的FA/FR/ED/CR、覆盖、分状态连续段，以及同数/同CR simple和score；同分不以GT挑样本，给同数FA上下界、同CR精确资格。长错误段定义>=5个相邻观测（<=3短、4单列），GT方向/段长只诊断，持续平稳的错误不解释为正确。

必要继续条件：完整VAL各组正确保留>=95%、全体同数/同CR严格优于simple和score、逐域/视频FA不增、连续正确FR不增；失败停止，不训练新网络、调门限、进入TEST或改正式policy。通过也只代表存在证据，另立设计才可能部署。历史充分子集给AUROC/同原接受数/同原CR/95%正确保留对照，不把离线GT工作点当在线策略。

清理只针对明确列举的work_dirs/port_reliability_oof_v1旧run内PT/PTH及字节相同的重复Git diff，以及work_dirs根目录9个明确列举的OOF同步bundle/压缩包；不删整个目录、不跟随软链接、不操作原数据/正式模型/其他候选。所有待删文件先记录相对绝对路径、大小、SHA；删除前全部验证。重复diff保留一个同SHA副本；包成员安全解包到本次cleanup证据目录并逐成员SHA核验后才可删包；bundle需确认commit对象已在Git。不满足验证者保留。报告/日志/源快照/配置/成员身份全部保留。用户本地既有交接草稿和未提交snapshot检查器保留，服务器既有worktree/index指纹不改。


### 45.2 实际运行、独立复算和停止结论

代码main f0e4929提交推送后在服务器原隔离分支只同步10个新增文件，原worktree/index指纹保持。首个增量bundle先决d2f450d在服务器缺失，任何源码同步/运行前即停止；改用已有931b98d作先决后核验成功，不改变源码。原路径mmrotljj、CPU运行RUN_ID=20261013_temporal_cpu_v1，9项时序检查通过。审查补齐了已知seq12拼接点的连续段统计（历史计算已重置），source main36b836e/server6ec0262，RUN_ID=20261013_temporal_cpu_v1_r2，10项时序测试通过；评分公式、窗口和门限全不变，VAL结果与首版相同。清理安全3项检查两端通过，共13项不同检查。

TRAIN门限固定0.0785511443740936；历史支持TRAIN2528/2558、VAL874/887。TRAIN拟合保护只针对原simple已接受的正确观测，seq05/seq06该支持为0，因此没有这两视频的门限约束；不是完整TRAIN正确率95%保证，不用此工作点宣布泛化。完整VAL保留原13帧历史不足（含1漏检），真实有输出12帧维持旧标志。

| 完整VAL | 接受 | FA | FR | ED | CR |
|---|---:|---:|---:|---:|---:|
| 原正式simple |843|190|10|33|653|
| 时序veto诊断 |785|155|33|68|630|
| 同785接受simple |785|170|48|53|615|
| 同785接受score |785|168|46|55|617|

同CR630下FA为候选155/simple178/score179；公平比较有局部增量，不能简单归为只靠拒绝更多。相对原policy拒错增加35，正确误拒增加23。整体正确保留95.02%，Real88.94%，seq07 51/63=80.95%，seq14 126/136=92.65%；Sim同CR453的FA39比simple34多5，同数差4，最长连续正确FR2比同CRscore1多1。分组同CR独立FA：Real116/139/136、seq07 104/117/128、seq14 12/12/13、Sim39/34/41（候选/simple/score），不相加成在线域路由。9个失败项含domain/sequence重复，不能当9个独立事件。

单独变化量在共同874输出的AUROC0.640887低于simple0.782019，seq07 0.544615 vs0.528533、Sim0.594360 vs0.692426；增量来自固定veto组合的局部互补，不宣布变化量整体更强。223个VAL尺寸错误有137个处于>=5观测长段；其中历史充分134个有100个变化低于门限。seq07长段114/162，历史充分111有82个变化低于门限，说明大量持续错误保持平稳。仅扩大窗口或平滑标志没有证据可以解决。

Mac/服务器独立复算3445行、135项状态/排序/AUROC，历史重放差0，源/policy/原框与三个原标志保持。修正版report SHAbba7e55a5196779f83df9f54b1e98a96964f230fadbd061027d06fa46ae5e058；结果集中work_dirs/port_reliability_temporal_evidence_v1/20261013_temporal_cpu_v1_r2/result，interpretation.md/json和双端review同目录。源码提交状态与结果来源分别记录，不混淆服务器隔离提交和main相同文件。CPU离线聚合约3秒，不是在线端到端时延。中心874/886、输出886/887、全帧正确中心874/887不变，训练更新0、检测推理0、TEST未读、无深度新结论。

**状态EVIDENCE_FAILED_STOP。** 不采用、不调窗口/评分/门限、不自动训练网络或跑TEST，正式M和policy保持。保留总体局部收益及分域失败，不把时序稳定当几何正确，也不由本项宣布所有时序方法无效。

### 45.3 已授权废弃辅助文件清理及保护核验

删除清单在首run cleanup_server集中：manifest.json、preservation_receipts.json、cleanup_receipt.json和protected_before/after.json。已在原服务器checkout核对无活动OOF训练；删除前全部候选SHA/大小/路径验证，再逐文件unlink复核。明确删除36个文件，5,100,416,266字节（4.7501328GiB）：15个放弃路线的PT/PTH、12个完全相同的重复Git diff、4个commit对象已保留在Git的同步bundle、5个安全解包并逐成员SHA核验后的旧包。全部日志/报告/配置/源码/数据成员身份保留；每种unique diff保留一份，不删除整个OOF目录、不跟随软链接。Mac扫描无辅助权重/重复项需要删除，空清理回执保存cleanup_mac。

正式B24 SHA8f8008…、midpoint SHA16c2fb…、policy SHA38d914…前后精确相同；服务器worktree/index保持b48cc0f92d424169abf2d767fd5d2ddd0e70974e14d3b92a363750d01addd6f1 / d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e。本地已将清单与36个执行回执逐条对账。**第44节中已删除辅助checkpoint的路径/SHA只作为历史来源，不能再用于续训/复现；删除基于用户本次放弃路线和清理授权，不改写旧失败事实。** 原数据、正式模型、其它候选与大小论文未删除或修改。

首回传包SHA5b1d554b67b8ad26187d6901e4449a317bd20a4ca8dd7f79f2dbe5a3dcf2dcc8，修正版包SHA6ef70ade5c57a5d519af7c87132f9c1c607fc1654cced080d36d3a490dd97739；安全成员和本地逐帧复核完成。分析及记录同步后只清理本次两个包和三个同步bundle，解包结果及删除回执完整保留。


### 45.4 用户授权清空已放弃OOF结果目录（2026-10-13）

用户进一步明确不需要保留截图所示大量失败结果。依据本次授权，清理范围升级为本地/服务器各自work_dirs/port_reliability_oof_v1完整目录，包括所有旧run报告、日志、配置副本和数据软链接；仅保留本交接中的历史结论与来源记录，不新增归档包。路径限定实际checkout内，顶层白名单、Git非跟踪及无活动任务核验后删除；递归不跟随软链接，原始图像/标注不删除，正式B24/midpoint/policy与其他已有Git改动前后验证不变。该目录及第44节对应原始结果路径此后不可再用作运行或逐帧复算输入，旧数值仍按历史记录解释。第45节时序结果及其清理回执是另一个目录，本次不扩展删除。

实际删除：本地13个顶层项、257个普通文件、44,419,735字节；服务器21个顶层项、427个普通文件、62,949个软链接项、270,856,053字节。软链接目标未触及，目录根已移除。没有新增结果备份或压缩包。


## 46. 因果趋势残差尺寸可靠性有限对照（2026-10-13）

状态EVIDENCE_FAILED_STOP，不采用本规则。固定M＝B24＋σ1.5/epoch03及匹配simple，完整TRAIN2558拟合一次、完整VAL887评价一次。没有TEST、检测推理、神经网络训练或深度新结果。旧门限0.0785511443740936、新门限0.10526070116970931，两者均按同一TRAIN程序得到。历史充分TRAIN2528、VAL874，暖启动/漏检完整报告；无框不进入FA/FR。TRAIN保护仅针对原simple已接受的正确输出，seq05/06支持为0，不是完整TRAIN95%保证。

| 完整VAL | 接受 | FA | FR | ED | CR |
|---|---:|---:|---:|---:|---:|
| 原simple工作点 |843|190|10|33|653|
| 旧中位尺寸偏差 |785|155|33|68|630|
| 新趋势预测残差 |778|155|40|68|623|
| 同778接受simple |778|167|52|56|611|
| 同778接受score |778|166|51|57|612|

新规则同接受数仍优于simple/score，FA分别少12/11，但不优于旧规则：同CR623旧FA153、新FA155；同CR630旧FA155、新FA158；同FA155的新规则最多保留624个正确框，旧工作点630。上述匹配门限均仅离线GT比较，不作在线policy。新规则正确保留93.97%，Real85.43%、seq07 45/63=71.43%、seq14 125/136=91.91%，Sim453/464=97.63%。Sim同CR453新FA44、旧39、simple34、score41；不能拿总体收益掩盖分域代价。18个失败项含域/视频重复，不能计为18个独立事件。

相比旧时序，新规则恢复17个正确框、又新增24个正确误拒；新检出16个错误，也重新接受16个旧检出的错误。TRAIN新旧FA均12，正确保留1410→1411，变化很小；Real FA11→12，不能用总体TRAIN统计掩盖域差异。新异常量共同874输出AUROC0.717974，高于旧0.640887，但低于simple0.782019。seq14异常量AUROC0.378731→0.575871，固定CR126时新FA11、旧12；该局部改善不构成全局采用依据。

机制核对：过去三点等间隔时，设计的median斜率等于(c-a)/2，median外推等于1.5c-0.5a，中间点消去。因此按固定公式执行但不称抗异常稳健趋势。24个新增正确误拒中16个历史端点含尺寸错误；24个的GT趋势残差均低于新TRAIN门限。16个新增FA中14个历史端点含错误。这支持“历史预测误差可污染趋势”的解释，但没有隔离唯一因果；GT仅离线诊断，不能据此在线选择历史。134个有历史的长段错误仍有99个变化低于新门限，旧为100，持续稳定错误未被解决。seq14最长连续正确FR由1→2，总体及Sim仍2。

实现只替换异常量；过去3次历史、断帧/漏检/分辨率/视频/seq12拼接重置、规范长短边和原标志回退保持。当前观测不参与自己的趋势预测。原中心874/886、输出886/887、全帧正确中心874/887保持；GT/未来/域身份不进在线API。源代码审核、两端各14项新增及10项旧测试通过。两端分别独立复算3445行、360项汇总/排序/AUROC/连续段检查，独立端点公式与实现最大差1.78e-15。服务器核验B/midpoint/policy及原始数据SHA前后相同；Mac没有两份前端权重，只核验本地拥有的输入/源码、独立复算及服务器固定身份回执，没有本地模型推理。CPU聚合10.50秒不是部署时延。

源码main86b156c，服务器d247566，源文件SHA相同。GitHub推送成功，服务器拉取被失效代理阻挡，使用核验后的增量Git bundle同步同一Git对象，只提交7个新增文件，服务器既有worktree/index指纹不变。未覆写原交接文档前缀。

建议：封存这版三点外推veto，保留新异常量的排序线索与旧时序公平比较收益。若另获授权继续，下一设计需明确如何防止错误历史把当前正确框判错，且在线不能使用GT历史资格。当前不调窗口、门限、不扩大网络或进入TEST；本轮不把趋势规则失败归为所有时序方法无效。

集中结果：`work_dirs/port_reliability_temporal_trend_v1/20261013_trend_cpu_v1/result`，含report、两臂门限、TRAIN/VAL逐帧、双端review及interpretation。结果报告SHA`20243ba7a8ec95f9d4b8bbc0bdef3e18bdefc34f083e19f4585308bc2c58a486`。失败版本只记本运行记录，不写论文主要结果。

临时文件清理：回传包SHA04c843ce…经双端核验后，本次两端结果压缩包和增量Git bundle已清理；服务器另清理本次同步脚本和记录传输临时文件。集中解包结果及回执保留，不清理其它候选或正式权重。


## 47. 历史线性一致性门控有限对照（2026-10-13）

状态EVIDENCE_FAILED_STOP，不采用本候选，按合同停止继续修补这套三观测切换规则。正式M＝B24＋σ1.5/epoch03及匹配simple保持。完整TRAIN2558拟合一次、完整VAL887评价一次；没有检测推理、神经网络训练、TEST或深度新评价。

唯一新机制：过去三个规范log长/短边的中间点偏离两端连线，q=max_axis|b−(a+c)/2|；q≤0.06327793458069397时用冻结趋势残差规则，否则用冻结中位尺寸偏差规则。两套拒绝门限0.10526070116970931及0.0785511443740936保持。归一化风险在门限1判断，仅为跨分支离线排序，不是错误概率。当前框不参与q，GT/未来/域/视频身份不进入在线判断；视频/断帧/漏检/分辨率/seq12采集拼接均重置，无框只记MISSING。

门控仅在TRAIN拟合：当前框正确且原simple接受、过去三个框也全部尺寸正确的1403个历史支持（Real662/Sim741），各组95%全同分q分位取最大值，seq12约束。seq05/06支持0明确披露；95%描述参考样本线性程度，不是完整TRAIN正确保留保证，也不能把历史一致当尺寸正确。两套旧门限按原TRAIN程序复算核验相等，没有重新选择或覆盖旧policy。

| 完整VAL固定工作点 | 接受 | FA | FR | ED | CR |
|---|---:|---:|---:|---:|---:|
| 原simple |843|190|10|33|653|
| 旧中位尺寸偏差 |785|155|33|68|630|
| 无条件趋势残差 |778|155|40|68|623|
| 历史一致性门控 |786|160|37|63|626|
| 同786接受旧中位偏差 |786|155|32|68|631|
| 同786接受simple |786|171|48|52|615|
| 同786接受score |786|169|46|54|617|

门控仍比simple/score有同数收益，FA少11/9，但未超过旧时序：同CR630门控FA164、旧155；同CR626门控160、旧153；同FA155门控最多CR619、旧工作点630。匹配阈值全为GT离线比较，不用于部署。相比无条件趋势，门控恢复6个正确框、新增误拒3个；检出2个错误，却重新接受7个错误，不能只报CR净增3。相比旧中位偏差，恢复14个正确框、新增误拒18个，检出9个错误、重新接受14个错误。

正确保留总体94.42%、Real85.93%、seq07 46/63=73.02%、seq14 125/136=91.91%、Sim455/464=98.06%。Sim同CR455门控FA44、旧39、simple34、score42；最长正确FR为1有所改善，但无法抵消FA退化。seq14同CR126门控FA11、旧12的局部信号保持，最长正确FR2仍高于旧1。16个门槛失败项含域/视频重复，不计为16个独立事件。

历史资格诊断：VAL874个历史充分观测中783个走趋势、91个回退中位规则；91个回退历史中74个含错误尺寸框，说明q有部分区分信息。然而全部328个含错误的历史中254个（77.44%）仍通过线性门控。seq07通过的154个历史中142个含错误（92.21%）；“线性一致”不足以证明历史可信。GT历史资格仅离线诊断，不转换为在线过滤标签。持续一致的错误与真实线性变化仍可有相同q。

TRAIN门控FA12/CR1415，旧FA12/CR1410，训练角色有小幅收益但未泛化。共同874输出门控风险AUROC0.708079，高于旧0.640887，低于无条件趋势0.717974及simple0.782019；不能用总体AUROC或TRAIN进步替代最终使用点验收。

工程核验：两端各12项新增测试通过，因果/原输入保持、GT隔离、分支边界、缺失/断帧重置、等比例及宽高角度规范、冻结两臂重放均检查。两端分别独立复算3445行及510项分组状态、排序、AUROC、连续段、分支资格，独立公式最大差1.69e−14。服务器原始数据、policy及正式B/midpoint SHA前后保持；Mac无两份前端权重，只核验可用输入/源码、固定身份与逐帧结果，未作本地模型推理。中心874/886、输出886/887、全帧中心正确覆盖874/887保持。CPU聚合13.43秒是离线分析耗时，不是在线部署时延。

源码main2883031，服务器dc3e60b，同一源文件SHA；本地GitHub推送成功，服务器Git代理失效后通过核验Git bundle同步同一对象，只提交7个新增文件，原worktree/index及交接前缀保持。不改论文主要结果文件。

建议：封存本门控及无条件趋势规则，不再调q、扩大窗口或增加切换条件，不进入TEST。保留旧时序对simple的公平比较增量作为未采用的研究证据；这次失败不代表全部时序方法无效。新的可靠性候选必须提供此前没有的正确性证据，单靠同一批预测的平稳/线性不能承诺识别持续错误。

集中结果：`work_dirs/port_reliability_temporal_history_gate_v1/20261013_history_gate_cpu_v1/result`，含报告、门控及两臂门限、TRAIN/VAL逐帧、双端review与interpretation。报告SHA`51f68e1409a78214f8c5557ad41d123ff01fff52e70ec5fe6a2c37fdec43fc55`。

临时文件清理：本次回传包SHAf9c0b172…完成双端核验后，两端结果包与增量Git bundle已删除；服务器本次传输脚本和记录片段临时文件也清理。集中结果及回执保留，其它实验、正式权重、policy和既有Git改动保持。

## 48. 原始时序偏差的直接监督融合有限对照（2026-10-13）

状态EVIDENCE_FAILED_STOP，不采用。已完成本地修改、代码审核、Git同步、指定服务器项目目录/mmrotljj CPU运行、回传和双端独立复算。正式M＝B24＋σ1.5/epoch03及匹配simple保持；无检测推理、GPU/神经网络训练、TEST或深度新评价。

唯一增量为原三个静态描述符加最初过去3个log长/短边的中位偏差，直接以当前主框双边是否均≤10%监督。固定class-balanced logistic，l2=0.1、Newton最多100步、梯度阈值1e-8。完整TRAIN2558含71错误；两臂共同历史充分2528仍含全部71错误，身份、标签、权重规则和预算相同。另拟合完整TRAIN3维模型作为两臂共同静态回退。没有新fit/probe角色、趋势、q门控、交互项或辅助检测器。GT只离线监督/评价，在线仅预测、原图大小、采样索引和重置。

两臂模型在VAL特征和门限计算前保存。完整已暴露VAL887各拟合一个全局门限：总体/域/视频正确样本95%全同分分位取最大值。候选0.5651616450647643，3维0.43162769069662676；标准化仅TRAIN。95%是VAL开发约束而非泛化保证，balanced sigmoid不称校准错误概率。历史不足共享静态风险但使用各臂门限，无框仅MISSING，不伪造时序零偏差。

|完整VAL工作点|接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|---:|
|4维候选|846|188|5|35|658|
|同TRAIN3维，自身组95门限|855|200|8|23|655|
|正式simple|843|190|10|33|653|
|同846接受3维/simple|846|193|10|30|653|
|同846接受score|846|196|13|27|650|

同CR658：候选FA188，3维/simple202、score199。与旧硬拒绝规则比较使用双方共同可达CR=min(候选CR,原simple最大正确保留数)：总体CR653，候选FA182、旧181；seq07 CR60，候选129、旧122。匹配门限只作GT离线诊断，不写policy；不因旧规则不能恢复原simple拒绝框而误判候选。

|VAL域/视频|FA|FR|ED|CR|正确保留|同数3维/simple/score FA|
|---|---:|---:|---:|---:|---:|---|
|Real|142|4|33|195|97.99%|148/148/146|
|seq07|129|3|33|60|95.24%|130/130/133|
|seq14|13|1|0|135|99.26%|13/13/13|
|Sim/seq10|46|1|2|463|99.78%|45/45/45|

总体正确保留658/663=99.25%，接受后错误188/846=22.22%；Real142/337=42.14%，seq07129/189=68.25%，seq1413/148=8.78%，Sim46/509=9.04%。全帧接受846/887=95.38%、正确尺寸覆盖658/887=74.18%。最长正确FR2，3维/simple同数和同CR为1；Sim候选最长FR1而同数控制0。95%保护均达标，但排序/连续性/超过旧时序门槛失败；40个失败条目含域/视频/控制重复，不计为40个独立事件。

TRAIN固定VAL门限后候选FA26/FR445/ED45/CR2042；同数3维/simple FA31、score53。TRAIN总体AUROC .801730→.831428，VAL总体 .785883→.759992；Real VAL .544925→.618119，Sim .694190→.666128，seq07 .539976→.553988。这里统一在本轮完整输出集合比较，不混用旧874个历史充分输出的AUROC。时序标准化权重+0.446432说明特征被使用，不证明泛化；VAL95%不能冒充TRAIN保护。

保存结果诊断：seq07三个FR（00008、00009、00216）的过去3帧均含2个尺寸错误，预测偏差 .15227/.13043/.13409，GT序列偏差 .01594/.01794/.02375。seq14_00046及sim_seq10_00326历史三个框都正确也误拒，说明门槛内预测波动也参与。上述是离线关联，不是唯一根因证明。联合监督提高正确保留，但未解决历史误差及域间失配。

两端最终15项测试通过，因果/GT隔离、缺失/断帧/分辨率/拼接重置、宽高/π周期/等比例及输出保护检查完成。独立复算3445行、257项模型凸目标驻点、门限、四状态/覆盖、同数/同CR、同分界、AUROC和连续段一致，历史公式差0。服务器正式权重SHA核验，本地缺两份正式权重如实披露；中心874/886、输出886/887、全帧中心正确874/887保持，方向标志、框和score保持。

初始审核的旧函数参数及JSON键排序导致的失败列表顺序比较已修复，增加回归测试；只重跑审核，不重新拟合/校准、不改原report。原审核/测试源码和两次修复回执集中保留。报告SHA1c6aba1cffddc002b627b2c4c3a5250780738541cf0acfe9db6129660441e29a。模型源码main3caba2b（服务器c3bebf0），审核最终mainf87ca8a（服务器98a1c09），同文件SHA。服务器Git代理故障后用核验增量bundle同步，原分支和其他worktree/index保持。

建议：按合同封存本次4维线性融合，不追加权重、门限、窗口、趋势、特征搜索或TEST。保留最初时序同数增量及本轮正确保留改善，两者均未采用；不把本轮失败扩大为所有时序方法无效。未采用结果仅记运行记录，不写论文主要结果。

集中结果：work_dirs/port_reliability_temporal_fusion_v1/20261013_fusion_cpu_v1/result，含模型、校准、原report、逐帧、双端review、解释及修复/清理回执。

临时文件清理：本次结果包SHA96a9c1bf…在双端核验和记录后删除；两端本次三个增量Git bundle及服务器传输脚本/片段临时文件清理。集中结果、正式权重/policy及其他实验保留。清理回执保存在本次result；无通配或目录批量删除。

## 49. 低时序偏差错误的条件可分性检验（2026-10-13）

状态：NO_SUFFICIENT_INTERACTION_EVIDENCE。仅基于冻结缓存完成CPU描述性检验，不拟合新评分、不重选门限、不训练交互候选、不运行检测或TEST。正式M、simple、中心/方向/输出身份保持。

目的：检验三个静态特征在同视频、相近预测尺寸下是否还能区分低时序偏差错误；核对原时序偏差与静态特征的条件关系。不是新候选的性能实验。

合同：完整TRAIN2558确定三个静态特征25/50/75%分位点及描述性风险方向，保存后才读取完整已暴露VAL887。低/高偏差沿用原TRAIN门限0.0785511443740936；历史不足独立列项，不填零。相近尺寸匹配限定同视频、同低偏差组，relative-log-size距离≤0.1，选最近正确框；对照可复用，不宣称独立配对。

实施前固定描述性继续条件：每个支持视频至少5个错误/5个正确；同一静态特征的TRAIN定向低偏差AUC≥0.6在至少2个TRAIN视频及1个VAL视频成立；静态TRAIN中位点×原时序低/高四格各有≥5错误/5正确，交互误差率差中差绝对值≥0.1且符号在至少2个TRAIN及1个VAL视频一致。两个条件需同一特征成立。该保守筛查不是统计显著性或模型性能门槛，失败不证明全部非线性信息不存在。

|当前冻结四维候选工作点|FA|FR|ED|CR|低/高/无历史FA|
|---|---:|---:|---:|---:|---|
|TRAIN|26|445|45|2042|25/1/0|
|VAL|188|5|35|658|164/19/5|

本轮不改变上述指标。VAL188个FA中164个（87.23%）低于原时序门限。低偏差组全部错误165个、正确632个，不把“全部低偏差错误”和“被接受的低偏差错误”混为一组；TRAIN相应56错误、2316正确。

|低偏差组，TRAIN固定风险方向|score AUC|相对尺寸AUC|长宽比AUC|
|---|---:|---:|---:|
|TRAIN总体|0.547|0.571|0.795|
|VAL总体|0.645|0.421|0.697|
|Real VAL|0.628|0.497|0.274|
|seq07 VAL|0.410|0.564|0.544|

|同视频、相近尺寸的低偏差错误—正确配对|匹配错误数|唯一正确对照|score方向一致比例|长宽比方向一致比例|
|---|---:|---:|---:|---:|
|TRAIN|56|55|51.79%|55.36%|
|VAL|164|73|50.61%|45.73%|

配对比例为错误框的特征风险方向高于正确对照的比例，同值计半；不是训练候选的准确率。VAL165个低偏差错误匹配164个，复用73个正确对照；不能将164个配对误写成164个FA或独立样本。

事实：TRAIN总体长宽比AUC0.795，Real VAL为0.274；同视频/相近尺寸后TRAIN、VAL长宽比方向一致比例分别55.36%和45.73%，score约51.79%和50.61%。各静态特征没有满足预定的跨视频支持条件。TRAIN高偏差错误仅15个，进一步四格分组后多数条件缺少错误支持；没有任一交互达到跨视频继续条件。

推断：总体可分性可能混入视频/尺度分布差异，静态风险方向跨视频失配；原四维信息尚无充分证据支持直接增加交互项。上述单变量及固定四格分析不能排除其他联合关系，也不能证明特征能力上限或唯一根因。

建议：本次检验到此停止，不自动训练交互候选，不继续调整窗口、分位点、支持数或筛查标准来获得通过。保留原四维同接受数的局部收益，正式simple保持。下一项需提出可核验的新增主框尺寸正确性证据，不能只增加同一四维输入的容量；尚未指定新结构。

工程：两端最终17项测试；本地回传独立复核3445行、3460项检查。服务器正式权重及所有输入SHA验证；本地缺两份正式权重明确披露。两端数学库造成272处浮点差，最大8.88e-16，仅审核增加1e-13绝对容差，计数/身份/分组严格一致；原报告、TRAIN规则及原审核源码保留，不重新拟合或分组。

来源：原服务器报告SHA d6dd51b1a121f7f3b7e8f417897715a227332d29f5ff629d80b437ea41acdbfd。代码本地main d022de5，服务器原分支3434767；最终审核本地c926f86。GitHub推送完成，服务器失效代理以核验增量Git bundle导入仅本次文件，保留其他工作区及index改动。

集中结果：work_dirs/port_reliability_conditional_evidence_v1/20261013_conditional_cpu_v1/result，含report、TRAIN_rules、双端审核、原审核/测试源码及修复回执。失败检验仅记运行记录，不改论文主要结果。

清理：完成回传、复核和记录后，仅删除本次双端结果压缩包及三个本次增量Git bundle；完整结果、历史实验、正式权重/policy保留。清理另存精确路径与SHA回执。

## 50. 冻结四维评分的单参数尺寸—时序交互对照（2026-10-13）

状态：EVIDENCE_FAILED_STOP，不采用、不追加其他交互项或预算。已完成本地实现/审核、GitHub推送、服务器指定项目/mmrotljj CPU运行及回传独立复核；未运行检测推理、神经网络训练、TEST或深度评价。正式M/simple保持。

研究决定更正：第49节要求至少2个TRAIN视频的四格各≥5错误，意味着至少20个高时序偏差错误，但全部TRAIN仅15个，原继续条件不可满足。因此它不能用于否定交互候选；历史原报告/状态保留。本次另立单参数探索合同，依据TRAIN总体相对尺寸条件赔率2.49/7.80、seq05 1.24/1.93，以及固定四维模型处交互零系数梯度−0.018666，六个视频梯度贡献同向（不冒充跨视频泛化）。VAL已有暴露，不能作为独立证据。

唯一增量：logit_new=logit_frozen4+gamma*z_relative_size*z_temporal。旧四维权重/截距/均值/标准差及共享静态回退冻结；不重新标准化乘积，不添加其他交互。gamma=0直接返回旧在线风险，精确中性身份；无框为MISSING/riskNone，历史不足沿用旧fallback，无伪造零偏差。在线仅预测/原图尺寸/采样索引/重置，GT只监督与离线评价。

训练：完整TRAIN2558中沿用相同2528个历史充分输出、71错误/2457正确；只训练gamma，从0初始化，类别平衡权重每类总量0.5，L2=0.1、Newton最多100步、梯度阈值1e-8。gamma保存后才解析/评分完整VAL887，单次选择总体/域/视频正确保留95%的全局最大分位门限。控制沿用原权重，不重拟合。

拟合完成：gamma=0.093269680983，3步；目标0.522826388→0.521954846，最终梯度8.22e-10。新门限0.629960043448仍由seq07保护60/63决定，旧四维门限0.565161645065。TRAIN目标下降只是拟合结果。

|完整VAL各自冻结工作点|接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|---:|
|原四维|846|188|5|35|658|
|单参数候选|854|194|3|29|660|

|同854接受比较|FA|FR|ED|CR|
|---|---:|---:|---:|---:|
|单参数候选|194|3|29|660|
|原四维|194|3|29|660|
|simple/同TRAIN静态3维|200|9|23|654|
|score|198|7|25|656|

同CR660：候选FA194，原四维193、simple202、score201。候选仍优于simple/score，但同接受数与原四维持平、同正确保留反而多1个FA，新增项没有形成增量。

|VAL域/视频候选|接受|FA|FR|ED|CR|正确保留|同CR原四维FA|
|---|---:|---:|---:|---:|---:|---:|---:|
|Real|344|148|3|27|196|98.49%|146|
|seq07|195|135|3|27|60|95.24%|129|
|seq14|149|13|0|0|136|100.00%|13|
|Sim/seq10|510|46|0|2|464|100.00%|46|

seq07同CR60时，FA129→135（+6），ED33→27（−6），3个原正确误拒保持。各自工作点10个标志变化：seq07新增接受7个错误、拒绝1个原错误接受，seq14/sim各恢复1个正确框；净FA+6、CR+2。这里同时包含评分与按合同校准的工作点变化，不将其全部归因于系数或门限。

剩余FA按原偏差门限：旧四维164低/19高/5历史不足→候选165低/24高/5历史不足，没有减少主要的低偏差错误接受。总体正确保留660/663=99.55%，接受后错误194/854=22.72%，错误检出29/223=13.00%；Real接受后错误43.02%，seq07为69.23%。最长正确FR2，原四维同数/同CR也为2；但simple匹配控制为1。

TRAIN AUROC原四维0.83143→候选0.83344，但同2227接受候选FA34/FR294，原四维33/293；VAL AUROC0.75999→0.75194，Real0.61812→0.61723，Sim0.66613→0.65216。训练loss/总体AUROC的小幅收益不能替代受保护使用点上的错误排序。

事实：单参数按预定目标正常收敛，无法优于原四维的目标工作点，seq07退化。推断：TRAIN残差信号未转化为所需的跨视频尺寸错误辨识；尚未确定唯一原因，也不证明所有交互模型无效。本次已完成应有的有限检验，应停止此项，不追加score/aspect交互、正则搜索或预算。保留原四维局部收益及正式simple，不自动迁移前端或运行TEST。

核验：两端最终18项测试通过，3445行、3580项独立核验；标量凸目标梯度、零更新还原、原基线风险/框/中心/方向/输出身份、GT/未来隔离、缺失/断帧/分辨率/采集拼接重置、宽高/π周期/等比例、单次门限、同数/同CR、同分边界/连续段均核对。服务器正式权重及输入SHA核验；本地缺两份权重如实披露。中心874/886输出、输出886/887、全帧中心正确874/887保持。

审核修复仅涉及JSON直方图整数键转字符串、跨平台历史浮点1e-13绝对容差；帧身份/计数严格一致。初始审核源码、测试及两次修复回执保留；不重新拟合、不改评分/门限/原报告。原报告SHA aa9d868dfa8116d21611531c422913acb565d2cefc6c4fd69aaaf4d0842aa3e6。

来源：模型实现本地main cb53e4d，服务器原分支26cbfba；最终审核本地788c9ef。GitHub推送完成，服务器代理失效以核验增量Git bundle导入仅本次文件；其他index/工作区/分支保持。

集中结果：work_dirs/port_reliability_size_temporal_interaction_v1/20261013_scalar_cpu_v1/result。只更新运行记录，不写论文主要结果；完整报告、逐帧、模型/校准、双端审核及分析保留。

清理：双端核验和记录完成后，仅按精确路径/SHA删除本次结果压缩包及三个Git传输bundle。清理回执留在result，其他实验、模型、正式policy保留。


## 51. 高正确保留区域的部分AUC训练目标对照（2026-10-13）

状态：EVIDENCE_FAILED_STOP，不采用本候选。完成本地修改/代码审核、服务器指定项目/mmrotljj CPU拟合、完整VAL、回传独立核验和记录。GitHub推送被自动审批拒绝：未明确授权具体origin仓库外发；代码本地提交4e44b16，服务器a3c2da0以核验增量Git bundle导入。服务器Git代理127.0.0.1:17890仍拒绝连接，没有成功普通pull，不称两端同一分支/提交。原文档草稿、服务器既有index/工作区保持。

研究问题：现有整体类别平衡BCE拟合/AUROC与高正确保留工作点不同，检验只改变训练目标是否提高原四维排序。它不是已确定根因；原时序变化平稳不能说明尺寸正确，换目标不增加图像证据。

实现：沿用原四维score-logit、relative-log-size、log-aspect、过去3个规范log边中位偏差；TRAIN均值/标准差冻结。错误尺寸为阳性，正确尺寸为阴性，FPR=FR/(FR+CR)，TPR=ED/(ED+FA)，正确保留95%对应FPR≤.05。候选是平滑上尾成对logistic代理，不是TPAMI PAUCI完整复现。

公式：h_j=mean_wrong softplus(w·(z_good_j−z_wrong_i)); J=tau+T/(beta*n_good)*sum_j softplus((h_j−tau)/T)+l2/2||w||²。beta=.05,T=.1,l2=.1，四个在线斜率从0初始化，tau从零斜率时的最优值初始化；Newton最多100步，梯度阈值1e−8，固定30次回溯。tau仅训练求解，不进入在线。排序截距不可辨识，因此冻结原BCE截距以维持共享静态回退的参照；这一做法不保证两支绝对风险分布对齐。sigmoid是排序风险，非校准错误概率。平滑代理有偏，优化它不保证精确pAUC/端点/逐视频收益。

完整TRAIN2558中共同历史充分2528，71错误/2457正确。BCE控制从0重拟合一次，复现原四维权重；候选另拟合一次，两臂各最多100步。控制和候选在解析VAL前分别保存。完整已暴露VAL887单次选择总体/域/视频95%正确保留最大分位全局门限0.4094748402105834，由seq07的60/63保护决定。GT只监督/离线评价，在线无GT/未来/视频域身份特征；历史不足原共享fallback3，无框MISSING/riskNone，断帧/漏检/分辨率/采集拼接重置。前端、框、score、数量、中心及方向标志保持。未进行检测推理、神经网络训练、TEST或深度新评价。

候选7步收敛，代理目标1.090177667→1.015149707，最大梯度8.88e−16。支持TRAIN精确归一化pAUC(.05)原四维.178031→候选.191904；完整TRAIN含静态回退后却.179451→.170651。前者不代表交付全输出收益。tau=.9045151；平滑尾部总权重1，最高5%正确样本仅占25.95%权重，有效正确样本数约944/2457。此诊断说明代理是软尾部，不能描述成只训练最难5%，也不据此重选T。

|完整VAL各自工作点|接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|---:|
|原四维|846|188|5|35|658|
|部分AUC候选|845|186|4|37|659|

|同845接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|
|候选|186|4|37|659|
|原四维|188|6|35|657|
|simple/同TRAIN静态3维|192|10|31|653|
|score|196|14|27|649|

同CR659：候选FA186、原四维189、simple202、score199。它在总体同数与同CR均有增量，不能写成完全无效。完整VAL AUROC原四维.759992→.787310，精确pAUC(.05).210160→.225331；这些描述性改善仍不等于正式采用。

|VAL域/视频候选|接受|FA|FR|ED|CR|正确保留|同CR原四维/simple/score FA|
|---|---:|---:|---:|---:|---:|---:|---|
|Real|335|140|4|35|195|97.99%|142/156/155|
|seq07|187|127|3|35|60|95.24%|129/130/144|
|seq14|148|13|1|0|135|99.26%|13/13/13|
|Sim/seq10|510|46|0|2|464|100%|46/45/44|

总体接受后错误186/845=22.01%，错误检出37/223=16.59%，正确保留659/663=99.40%，全帧接受845/887=95.26%，正确尺寸覆盖659/887=74.30%。Real错误接受占140/335=41.79%，seq07仍127/187=67.91%。原低/高/无历史FA164/19/5→164/20/2，主要低偏差错误接受没有减少。

未通过原因：Sim同CR464时FA46，比simple45/score44差；seq07正确FR最长连续2帧，匹配simple和同CRscore最长1帧。候选四个FR为seq07_00008/00009/00216及seq14_00018，三帧seq07正确误拒与旧四维相同。没有通过放宽连续性门槛或加入恢复规则采用。13个失败条目含域/视频/控制重复，不计为13个独立失败。95%各视频保护满足，不把此版写成覆盖保护失败。

TRAIN应用VAL门限FA22/FR622/ED49/CR1865，正确保留约74.99%；同1887接受原四维FA21/FR621，比候选好1个。VAL95%不冒充TRAIN保护。TRAIN总体AUROC.831428→.824479，而VAL上升，不能简单归纳为fit成功/泛化失败，也不能只看损失替代实际交付结果。

事实：该固定训练目标有VAL总体与seq07小幅排序增量，但仍未满足逐域/连续正确覆盖合同。推断：目标选择可以影响可用排序，但难以凭同四维信息识别大量稳定错误；尚未确定唯一根因或能力上限。建议保留研究信号，正式simple保持；封存本次目标，不自动搜索beta/temperature/l2、补其他交互/窗口/趋势或运行TEST。后续若另提方案，应先解释新增信息或明确新机制，而非把小幅收益当作继续无界调参理由。

工程：两端各13项测试通过，3445行及3655项独立核验一致；BCE控制驻点梯度1.97e−12、候选8.88e−16，独立评分最大差1.11e−16。检查解析梯度/Hessian、pAUC同分截断、身份/因果/GT隔离、冻结控制及缺失回退、在线几何和三标志保护。服务器前端权重SHA核验，本地缺两份权重如实披露。中心874/886输出，输出886/887，全帧中心正确874/887保持。审核没有修复/重拟合/重选门限，原报告保持。

报告SHA f888c52163363caf21fc5165e7bc383d9381d961ee1d6bbebb97c10f59b0fcef。集中结果work_dirs/port_reliability_partial_auc_v1/20261013_pauc_cpu_v1/result，含原report/model/control_fit/calibration、逐帧、两端review、mechanism_diagnostic及interpretation。仅本运行记录写失败候选，不改论文主要结果。

文献身份：借鉴 https://arxiv.org/abs/2512.01213 的部分AUC目标思想；https://jmlr.org/papers/v25/22-0795.html 的约束式错误取舍仅作任务参考，不引用其统计保证；https://proceedings.mlr.press/v244/kassem-sbeyti24a.html 的错误预算评价，不重做已失败的单纯门限实验。

清理：本次结果包SHA5b4af1ef…及两份增量Git bundle在双端复核和记录后按精确路径/SHA清理；完整结果目录与回执保留，不清理其他实验或正式模型。GitHub推送仍待具体目的地批准，不称完整GitHub同步完成。


## 52. 小型图像质量头按视频交叉拟合与时序融合（2026-10-13）

状态EVIDENCE_FAILED_STOP，不采用，不重选结构/折/种子/epoch/权重或门限。完成本地修改/代码审核、GitHub推送、服务器项目/mmrotljj CPU运行、回传独立复算与记录。源码main c7fffbe，服务器原分支05825af，同文件SHA；服务器代理失效，核验Git bundle仅导入8个新增文件，既有index/工作区保持。第51节GitHub阻断已由用户明确授权后补齐main核验，历史阻断事实保留。

目的：检验图像证据能否补充识别低时序偏差错误，并避免质量头自身TRAIN的完美评分让融合器过度依赖图像。不是已确定根因。固定M＝B24＋σ1.5/epoch03、正式simple/policy、部分AUC和旧四维。没有辅助检测器训练、重新检测推理、新特征提取、TEST或深度评价。

合同：完整TRAIN2558含71错误，内部按6个视频留一视频训练质量头，外部TRAIN/VAL/TEST身份不变。只复用原生B实际回归输入2304→既有TRAIN PCA256缓存。PCA/检测器/冻结时序头仍已见完整TRAIN；OOF仅隔离质量头拟合标签，不声称全链未见视频。每折259维标准化只用其他视频；另训完整TRAIN头供最终交付。

质量头沿用score锚点258→16→8→1 ReLU+beta，4290参数、FP32、seed1701；末层/beta零初始化，每头重新开始100epoch/batch256/类别平衡BCE/AdamW .001、weight decay1e−4只作用weight、clip5，无选epoch。正式6折+全量共7头、6300更新。单测另用2个合成小头验证排除，未复用为正式权重。留下seq05时拟合错误27个，其他折58～70；没有增加新错误材料。

两臂：A(in_sample_fusion)用完整TRAIN头在自身TRAIN的评分拟合；B(crossfit_fusion)用视频留出评分拟合，仅B可晋级。二者线上使用同一个最终完整TRAIN图像头。只读部分AUC及图像风险，固定logit截断±12，标准化只用各臂TRAIN；共同2528历史充分TRAIN仍71错误。logit_new=logit(r_partial)+delta0+delta1*z_partial+delta2*z_image，3残差参数从0初始化，中性身份精确；类别平衡BCE、l2=.1对两斜率、Newton100/tol1e−8/回溯30。没有交互、域或视频身份特征。OOF与全量风险不作事后对齐，记录失配。所有头和融合权重在解析完整VAL前保存。

无框MISSING/riskNone；历史不足保留旧部分AUC回退risk，但应用各臂新全局门限，risk保持不等于标志保持。GT仅监督和离线评价，在线只预测/原图大小/帧索引/实际原生B patch/reset。框、score、输出/漏检、中心/方向保持。部署候选仅增加一质量头/PCA和3参数融合，没有新增检测器；本次仅缓存实验，未完成端到端集成或时延测量。GPU2/3忙，使用CPU单线程，不干预其他任务。

完整已暴露VAL887一次选择每臂全局组95门限：A .28624435797437753，B .5317310050227115，均由seq07保护60/63决定。TRAIN交付诊断使用全量头，OOF只供B拟合，不把OOF伪装线上结果。

|完整VAL各自工作点|接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|---:|
|冻结部分AUC|845|186|4|37|659|
|A训练内控制|843|187|7|36|656|
|B交叉拟合候选|847|188|4|35|659|

|同847接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|
|B|188|4|35|659|
|A|190|6|33|657|
|部分AUC|187|3|36|660|
|原四维|189|5|34|658|
|simple|194|10|29|653|
|score|196|12|27|651|

同CR659：B FA188，A191、部分AUC186、原四维189、simple202、score199。B优于新控制，但未超过已有部分AUC。各自TRAIN交付点A FA0/FR23/ED71/CR2464，B FA18/FR576/ED53/CR1911，VAL95%不冒充TRAIN保护。

|VAL域/视频B|接受|FA|FR|ED|CR|同CR部分AUC/simple/score FA|
|---|---:|---:|---:|---:|---:|---|
|Real|337|142|4|33|195|140/156/155|
|seq07|189|129|3|33|60|127/130/144|
|seq14|148|13|1|0|135|13/13/13|
|Sim/seq10|510|46|0|2|464|45/45/44|

正确保留659/663=99.40%，接受后错误188/847=22.20%，错误检出35/223=15.70%，正确尺寸全帧覆盖659/887=74.30%。seq07接受后错误129/189=68.25%。各视频95%保护满足，最长连续正确FR2，比匹配simple及同CRscore的1差。23失败条目含域/视频/控制关联重复，不等于23独立失败。

关键新发现：图像TRAIN训练内AUROC1.0，视频留出总体.549103/Real .559696；seq05 .570120、seq06 .525896、seq12 .522059、seq13 .473785；Sim .828648只1个错误，不推广。B图像标准化残差系数.031154、A1.168500；B时序.688937、A .100741，两臂标准化不同，系数不解释为独立贡献或概率。全量−OOF logit差分位(0/5/50/95/100%)−12.8173/−3.2026/−.3696/.0097/13.5123，评分失配存在，尚非唯一根因。B VAL AUROC.788818略高于部分AUC.787310，受保护工作点仍多2FA。

交付拆分：低/高/无历史FA，B164/19/5，部分AUC164/20/2，主要低偏差FA数量未减少。相比部分AUC实际点只有4个标志变化、正确接受集合保持：real_seq07_00159由FA变ED；real_seq07_00001/00136/00137这3个历史不足错误由ED变FA。风险旧值回退但新门限重新接受，净FA+2=检出1/重新接受3；不能笼统说历史充分图像融合全部退化，也不能凭该1帧宣布解决稳定错误。本次未事后恢复旧回退门限或另立新候选，原权重/门限/报告保持。

结论：质量头训练内辨识不代表视频留出表现；交叉拟合减少了融合对图像评分的依赖，尚未形成有效新增辨识。回退risk与新门限配合造成额外交付损失。不能认定原生特征、PCA、标签、映射或损失为唯一根因。封存本候选，不扩质量头、不追加预算/种子/折数、不重启辅助检测器或自动TEST；正式simple及部分AUC研究信号保持。若另获授权继续，必须明确静态回退决策身份和稳定错误的新信息，不能靠恢复几帧回退宣布大幅降FA。

工程：本地13项通过/1项Torch跳过，服务器14项全部通过；两端独立复算3445行、3805项核验、6300正式更新。整视频标签排除/每帧唯一OOF/折内标准化/共同初始化/固定epoch和批次顺序、融合驻点/逐帧风险、门限/同数同CR/并列/AUROC/连续段一致。融合梯度B2.97e−11/A4.64e−14，独立风险最大差1.89e−15。审核无修复/重训；服务器前端SHA核验，本地缺两份权重如实披露。中心874/886输出、输出886/887、全帧正确中心874/887保持。

报告SHA7ecb136723d98955e45fd297decd3e991b36522e7f7b3208faa890272dea40fc。集中结果work_dirs/port_reliability_image_temporal_crossfit_v1/20261013_cached_crossfit_v1/result，含7头JSON、融合、OOF/训练内评分、完整逐帧、双端review、mechanism_diagnostic和interpretation；未带回旧特征缓存或检测权重。只更新运行记录，不改论文主要结果。

收尾：双端复核和记录后，仅按精确路径/SHA删除本次回传包60462759…和Git bundle f28c1bea…，保留完整结果和清理回执。GitHub main源码推送成功，服务器普通fetch失败、核验bundle同步及原分支身份准确披露。


## 53. 历史不足回退决策身份修正：仅本地实现（2026-10-13）

状态：LOCAL_IMPLEMENTED_NOT_RUN。用户本轮先授权既有服务器流程，随后明确改为“现在只去修改代码吧，不去服务器上运行了”。此后只进行本地实现、审核和必要检查；此前服务器仅只读核对项目路径/分支/状态，没有上传、提交、启动任务或删除文件。源码未提交/推送；保留原文档草稿、已有未跟踪脚本及服务器index/工作区。

版本port_reliability_fallback_decision_v1。唯一交付变化：历史充分沿用已冻结crossfit_fusion风险和门限0.5317310050227115；历史不足沿用partial_auc4回退风险和原门限0.4094748402105834，使原回退最终标志保持。不是新增图像证据、神经训练或门限搜索。无框riskNone且size_accepted=False；中心/方向/框/score/输出身份沿用原链。原crossfit/partialAUC源码、权重、门限和报告均不覆盖。

新增7个源码/合同文件：utils/port_reliability_fallback_decision_v1.py，tools/run_/review_对应Python，protocol/sources JSON，tests对应文件，以及根tools/run_port_reliability_fallback_decision_v1.sh。线上CausalFallbackDecision继承原图像时序API，只修改size_accepted的分支门限，输入无GT；历史不足不要求native patch。离线完整TRAIN/VAL工具输出逐帧显式布尔标志、分域/逐视频FA/FR/ED/CR/MISSING、正确保留/全帧覆盖、连续正确FR、原工作点及同数/同CR对照，四状态变化逐帧记账。不同分支门限不伪装为单一校准概率，不新增总体AUROC结论。

既有模型和calibration/report/逐帧/双端review SHA冻结；工具复核来源后才计算。未来运行零拟合、零校准、零检测推理/更新，只复算完整2558 TRAIN与887已暴露VAL。维持上一版严格各组95%正确保留、同数/同CR与连续FR门槛，失败即停止，不自动采用/TEST。设计已受VAL反馈影响，明确披露，不当作独立验证。

本地代码审核修复对照集合遗漏in_sample_fusion及独立gate列表排序问题；新增合成端到端review序列化/篡改检出测试。11项测试全部通过，包含不同分支/包含等号门限、风险数值、无GT/无输入修改、回退评分身份、等门限还原、缺框、历史间断/重置/分辨率、在线几何/score/中心/方向保护、合同SHA核验、指标/gate与独立审核。来源check-only、Bash语法及git diff --check通过。本地缺两份前端pth，check-only显式allow-absent，仅检查缓存/来源，未冒充实际前端权重核验。

未运行完整TRAIN/VAL性能评估、服务器任务、训练、TEST或深度评价，无新增FA/FR性能结论。此前FA185/CR659只是第52节逐帧变化的反事实预测，不记为本轮结果。未生成或清理传输包，没有修改论文主要结果或正式policy。


## 54. 回退决策身份修正：服务器完整复算与收尾（2026-10-13）

状态EVIDENCE_FAILED_STOP，正式M/simple/policy保持，不采用、不自动进入TEST。第53节仅本地状态是历史事实；本轮用户重新授权服务器既有流程，现已完成运行、回传及双端独立核验。源码GitHub main d27e7b0；服务器原codex/oof-gradient-chain-v1分支9b54b84。普通fetch因127.0.0.1:17890代理拒绝失败，核验增量Git bundle导入7个新增文件、仅提交这些文件；既有index/文档保护核验通过，不冒充普通pull或两端同HEAD。

唯一变化是历史不足使用原partial_auc4门限0.4094748402105834，历史充分仍用冻结crossfit_fusion门限0.5317310050227115。权重、评分、历史规则、中心/方向/框/score/数量/输出身份保持。服务器指定项目/mmrotljj，CPU单线程，无需GPU；零训练/拟合/校准/检测推理或更新，无TEST/深度评价。设计受已暴露VAL发现影响，如实披露。

|完整VAL实际工作点|接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|---:|
|部分AUC|845|186|4|37|659|
|原交叉拟合融合|847|188|4|35|659|
|仅回退决策修正|844|185|4|38|659|

|同844接受控制|FA|FR|ED|CR|
|---|---:|---:|---:|---:|
|候选|185|4|38|659|
|部分AUC|186|5|37|658|
|原交叉拟合融合|187|6|36|657|
|原四维|188|7|35|656|
|训练内融合控制|188|7|35|656|
|simple|191|10|32|653|
|score|195|14|28|649|

同CR659：候选FA185，部分AUC186、原交叉拟合188、原四维189、训练内融合191、simple202、score199；所有方法FR4，ED=223−FA。同数/同CR是离线比较，不把GT匹配门限写入线上。

|VAL候选域/视频|接受|FA|FR|ED|CR|正确保留|
|---|---:|---:|---:|---:|---:|---:|
|Real|334|139|4|36|195|97.99%|
|seq07|186|126|3|36|60|95.24%|
|seq14|148|13|1|0|135|99.26%|
|Sim/seq10|510|46|0|2|464|100%|

总体正确保留659/663=99.40%；接受后错误185/844=21.92%；错误检出38/223=17.04%；全帧接受844/887=95.15%；全帧正确尺寸覆盖659/887=74.30%。seq07错误接受126/186=67.74%，高FA未解决。无框1帧独立MISSING，不参与四状态或判别。中心874/886输出命中98.65%，输出886/887=99.89%，全帧正确中心874/887=98.53%，保持。

逐帧来源：较原融合只改变3个VAL标志，real_seq07_00001/00136/00137历史不足错误由FA→ED，正确接受集合一致。较部分AUC只real_seq07_00159历史充分错误FA→ED，收益仅1个。低/高/无历史FA164/19/2，原部分AUC164/20/2、原融合164/19/5。主要低偏差164个错误接受保持，没有新证据表明已改善稳定错误辨识。不将回退身份修正包装为图像头新增训练收益，也不宣称分支风险成为共同校准概率或新增总体AUROC。

完整TRAIN交付FA18/FR580/ED53/CR1907，接受1925/2558；正确保留1907/2487=76.68%，seq05只有43/516=8.33%，正确误拒473、最长40帧。较原融合FA保持，FR576→580，real_seq06_00000/00090/00091/00092这4个历史不足正确框恢复原部分AUC拒绝。回退身份恢复有TRAIN正确覆盖代价，不能称全数据单调改善；VAL的95%不冒充TRAIN保护。TRAIN使用完整质量头，OOF只用于上一版融合拟合，本轮不新增材料。

未通过：Sim同CR464时FA46，部分AUC/原融合/simple45、score44；seq07连续正确FR最长2帧，匹配simple及同CRscore最长1；seq14同CR135训练内控制FA12而候选13（其同数/连续性亦触发）。18个失败条目含域/视频/控制重复，非18个独立失败。各组95%正确保留通过，不写成VAL95失败。不放宽门槛补采用，不改窗口/恢复规则/门限、不追加网络或预算。

工程：上一轮本地11项测试通过，本轮服务器11项全部通过；双端独立复算3445行、3700项核验，历史充分旧融合决策/历史不足原部分AUC决策逐帧身份一致，指标/匹配/连续段/变化/门槛/来源核验通过。服务器实际前端SHA核验；本地缺2份pth仅显式allow-absent审核缓存身份，不冒充全前端现场核验。没有审核后改源码、重拟合或重选门限。

建议：保留这次回退决策机制修正和总体小幅研究信号，但不替换正式版；部分AUC仍是轻量后续对照。下一项不能继续围绕这1帧收益扩大质量头或改门限。若研究大幅降FA，需要先说明识别低偏差稳定错误的新证据，并同时保护正确覆盖；现有结果尚不足以指定有效新结构。

结果集中work_dirs/port_reliability_fallback_decision_v1/20261013_frozen_fallback_v1/result；report SHA64f49ecbeed092c178ffecefaebf6cae271c2192fb1a0ef9bd4ad6d3c83df08e。保留原report、逐帧、日志、双端review、interpretation与同步/记录/清理回执，仅运行记录写未采用实验，不修改论文主要结果。

收尾：双端复核/记录完成后仅按精确路径、非符号链接及SHA清理本次回传包work_dirs/port_reliability_fallback_decision_v1_analysis_20261013.tar.gz（SHA8b71ed49…）与增量bundle work_dirs/symEOOD_fallback_decision_v1.bundle（SHA908f3e9e…）。完整结果、7个代码文件、原模型/旧实验及其他数据保留；清理回执确认实际完成。


## 55. 同次前向候选尺寸证据：完整TRAIN前提检验（2026-10-14）

状态TRAIN_EVIDENCE_FAILED_STOP。本轮用户授权实现、审核、Git同步、服务器运行、回传分析、记录及本次传输包清理。已实现port_reliability_candidate_support_v1；未采用，正式B24＋midpoint σ1.5／epoch03、simple和policy保持。失败原因是预先规定的候选支持合同未满足，不能写成判断器训练后性能下降、候选信息完全无用或唯一根因已经确定。

机制：观察原B一次前向原解码器的实际返回局部变量。保留padding筛选和score_thr=.05，读取Top1之前的原生预测；按实际surviving source index排除原B主框，不用最近框匹配代替来源身份。不改变原Top1、框、score、数量、中心或方向。候选与最终M统一原图坐标、规范长短轴/π周期，中心在M框内且轴差≤15°，按score/原索引取最多8个；至少3个才读取两项median log(candidate_L/M_L)、median log(candidate_S/M_S)。候选不足/历史不足保留原partial_auc4决策与冻结门限.4094748402105834，不通过新门限改变回退标志。在线不读取GT、视频或域身份，缺框单列MISSING。

冻结前提：每域候选覆盖≥50%；同时具备原3帧历史及候选支持的真实TRAIN尺寸错误≥20；至少一项有符号差异在两个Real视频中按全TRAIN确定的同一方向AUC≥.55，各视频至少3个错误/20个正确。上述是本轮前瞻可行性常数，不是理论保证或VAL调参结果。通过后才按同TRAIN身份、同初始化、同部分AUC目标/预算拟合4维/6维轻量对照，保存模型后进入完整VAL；本轮未进入这一阶段。

|完整标准TRAIN|输出|至少3个有效邻居|有效覆盖|受支持尺寸错误／全部错误|
|---|---:|---:|---:|---:|
|Real|1810|171|9.45%|11／70|
|Sim|748|111|14.84%|0／1|
|总体|2558|282|11.02%|11／71|

同时具备历史与候选的拟合可用帧279（268正确、11错误）。两域覆盖、至少20个错误、两个Real视频信号条件均未通过；不是先训练判断器后用TRAIN性能拦截。seq05有7个受支持错误，seq06/seq12各2个，其他为0。受支持/历史子集的描述性AUC长边.7809、短边.6604；seq05 .7043/.7494（7错误），seq12 .88/.84（仅2错误），seq06 .5741/.4630（仅2错误）。只有seq05具备规定样本数，不能把小样本AUC写成已验证跨视频增量或方向一致性的证明。

额外事实：原score筛选后2558帧候选池分别有1/2/3/4/5/6/7个的帧数697/998/560/218/62/18/5。至少一个合格非自身邻居1838/2558=71.85%，其中覆盖50/71个错误。离线GT核对这些帧：主框错误且至少一个邻居尺寸正确36帧，主框错误且全部邻居错误14帧；主框正确但全部邻居错误48帧，主框正确且有正确邻居1740帧。这些邻居是同一个B的假设，不是新辅助检测器，也不能在线用GT挑选正确邻居。该统计仅说明1—2个候选仍可能含线索，同时存在误伤正确主框/共同错误风险；没有因此放宽本轮规则、重拟合或进入VAL。

工程：原B完整2558帧原图6分量回放最大差全部0；冻结参数/buffer摘要保持。最终M及原标志从SHA固定缓存复用，未冒充重新运行完整midpoint。图像/标注身份及两份权重在服务器核验；本地缺两份pth和原ROI缓存清单，仅显式allow-absent核对其他来源/回传证据。GT仅用于离线标签/诊断。完整运行零拟合、零检测器更新，无VAL推理/校准、无TEST或深度评价，因此没有新的FA/FR/ED/CR性能收益结果。

本地与服务器16项测试通过；测试及完整GPU回放覆盖来源索引排除、单候选原解码器rescale别名、旋转/尺度/宽高交换/π周期、缺框、原回退标志保护、GT无关输入不可变、4维目标还原、6维梯度/Hessian与独立站点检查，以及可选拟合→校准→评价支路。补充端到端支路检查发现FA分层统计缺少循环，修复并加入回归测试后重新运行；首轮部分提取以工程原因中止，未形成报告/权重/VAL结果，日志和工程中止记录保留。不把完整修复运行的2558次推理或274.79秒冒充包括首轮部分运行在内的总成本。计时含Python tracing、数据/来源核验，不是部署延迟或FPS。

双端独立核验2558行/2558项候选几何及来源检查；本地再独立核对45项覆盖/错误统计。report SHA2663a8bdbdb9d77fc1b627d5ebf939d681e569a55ac073f9fb473ed34ddfa443。完整结果统一保留work_dirs/port_reliability_candidate_support_v1/20261014_native_candidates_gpu3_fix1/result，含report、候选逐帧、TRAIN_screen、server/mac review、interpretation和核验回执；未写入论文主要结果。

Git：main源码226fde1（初始3fc4469＋评价修复），已推送；服务器保持原codex/oof-gradient-chain-v1分支3fa467e（初始e5b49e4＋修复）。普通fetch被127.0.0.1:17890代理拒绝，按SHA核验增量bundle只导入本实验8个文件，保护原暂存patch SHA d54105b5…和文档草稿；不冒充普通pull或两端同HEAD。使用mmrotljj和物理GPU3。

建议：保留局部TRAIN线索，但当前“至少3个候选”的广覆盖合同不成立，停止这一版本，不降低原读取门限凑候选、不自动调min_neighbors或开展新训练。若继续研究稀疏候选，应另立针对1—2个有噪声定位假设的设计并重新固定对照，不能把本次小样本AUC当作判断器收益。当前研究参考继续保留部分AUC，正式simple不替换。

收尾：分析和双端运行记录同步后，仅按精确路径/非符号链接/SHA清理本次回传包work_dirs/port_reliability_candidate_support_v1_analysis_20261014.tar.gz（4023dfc7018c4f8be1d41c88ddee267528edb7a566f15cb9365ba87a9bc337ed）及本实验source.bundle（fddb511a…）/source_fix.bundle（1ca992fe…）。完整逐帧gzip、结果目录、代码、旧实验与其他文件保留；实际清理完成以cleanup_receipt为准。运行说明原位追加，保留既有未提交文档内容，不整份覆盖或提交原草稿。


## 56. 稀疏原生候选对时序判断的增量：TRAIN缓存检验（2026-10-14）

状态DESCRIPTIVE_TRAIN_COMPLETE_NO_TRAINING_AUTHORIZED。按用户授权完成代码、审核、Git同步、服务器运行、回传核验、记录及本次传输包清理。此状态表示本轮执行器没有下一阶段训练职责；未自动进入训练/VAL/TEST。正式M/simple/policy、原时序/部分AUC/回退修正版保持。没有新增性能收益结论。

本轮以回退修正版为研究参照，部分AUC四维为轻量对照，复用同次B候选池与冻结M，不从simple重新起步。前两个邻居按原score/原索引排序，分别保留log长短边差；不降低候选score门限，不改原筛选/时序窗口/回退门限，不用GT选择邻居。完整2558帧TRAIN，不新增fit/probe角色。GT仅用于离线标签；缺框独立MISSING。

|原TRAIN研究版本|FA|FR|ED|CR|≥1邻居覆盖FA|≥3邻居覆盖FA|
|---|---:|---:|---:|---:|---:|---:|
|回退修正版|18|580|53|1907|12|1|
|部分AUC四维|22|622|49|1865|14|1|

回退修正版7个FA只有1邻居、4个有2邻居、1个有≥3邻居、6个无邻居。≥1邻居还覆盖1357个CR、431个FR、38个ED。候选覆盖不是错误检出，更不能把上轮覆盖50个尺寸错误写成可降低50个FA；原接受标志未改变。

已接受集合FA/CR之间，第一邻居长边/短边有符号差总体“较大值对应错误”AUC为.4455/.5114；第二邻居为.4283/.3461（只有5个FA有第二邻居）。按同视频/原风险无标签四分位层的配对AUC分别.3682/.5593/.3899/.3073。完整报告同时给出较小值方向=1−AUC，未选择最佳方向或扫描门限。分层仅是描述性检查，不证明条件独立或跨视频增量，配对数不能冒充独立样本数。

新发现：第一邻居短边差seq12总体/视频风险分层AUC .3688/.3793，seq13为.5935/.6077，两者各4个FA，方向相反；第二邻居短边差seq12 .9714只有1个FA，而seq06 .2596只有2个FA。第一邻居长边差部分Real有局部线索，但seq12 .5063近随机，Sim .9864只有1个FA且方向相反。两个邻居长边同偏小包含3个FA和268个CR，短边同偏小包含3个FA和196个CR。不能直接按候选分歧/偏小拒绝，也不能凭总体AUC宣布会保护正确框。

建议：目前不足以支持新的统一稀疏候选决策修正训练，不自动将min_neighbors改1重跑、不新增模型/扩大门限搜索。保留已有时序四维、部分AUC、回退决策身份修正，结束本次有限证据检验。尚不能认定原生候选完全无用或唯一根因已查明；若继续，需要明确处理跨视频方向不一致和候选共同错误的新证据机制，目前未指定有效结构。TRAIN大量FR及VAL高FA仍是既有版本限制，没有因此解决。

工程：本地/服务器11项测试通过，2558行/3035项两端独立核验一致。审核统一恰好10%误差边界公式并补测试。零拟合/校准、零检测推理/更新，未打开VAL/TEST，不重新声明前端权重现场核验。main源码1aebc64已推送；服务器普通fetch仍被127.0.0.1:17890代理拒绝，经SHA核验增量bundle仅导入7个新增文件，保留原codex/oof-gradient-chain-v1分支7d4b992。原暂存patch同步前后SHA d54105b5…保持，不冒充普通pull/两端同HEAD。mmrotljj、准确项目路径、CPU单线程；不占GPU。

结果统一work_dirs/port_reliability_sparse_candidate_v1/20261014_cached_sparse_v1/result，含report、2558行diagnostic_rows、解释、code_review、双端独立审核及收尾回执。报告SHA b1b933dde25720a094e0401aa8f3f056871805f69051c0420bd7e03961af64de。只更新运行记录，不修改论文主要结果。回传首次本地解压仅因系统Python版本不支持filter参数中止，改配置Python3.12、核验路径/非链接后完成；无结果丢失。

清理范围仅本次work_dirs/port_reliability_sparse_candidate_v1_20261014_cached_sparse_v1_analysis.tar.gz（ab2827664f9b8508aa6c2f20587bfaaa5eaafd2632c63b83a0f0fd491de76ce7）和work_dirs/port_reliability_sparse_candidate_v1/source_1aebc64.bundle（86f2392f64270a82b8c5dabd5f48f09e9f1b50e41bb99c7c9dca6daac31e5748）；准确路径/非链接/SHA核验后清理，实际完成以两端cleanup_receipt为准。完整结果、原candidate gzip、旧实验和其他文件保留。


## 57. 困难错误固定加权的部分AUC对照（2026-10-14）

状态EVIDENCE_FAILED_STOP。经检索此前没有本项实验；按用户授权完成代码、审核、Git推送、服务器运行、回传独立核验、分析记录与本次传输清理。没有采用、TEST或深度评价，正式M/simple/policy及原四维/部分AUC/回退修正保持。

唯一训练增量：h_good由全部错误均匀平均改为归一化固定q；完整TRAIN错误71个按冻结原部分AUC风险选最低25%共18个（边界同分全部纳入），困难2倍、其他1倍。不用VAL确定的FA名单、不动态挖掘、不搜比例或倍率。18个分布seq13/06/12/01/Sim为7/5/4/1/1。原四维、TRAIN标准化/截距、beta.05/T.1/l2.1/零初始化/Newton100保持。两臂相同2528个历史充分TRAIN、71错误/2457正确，硬集合先保存、两个模型保存后才解析完整已暴露VAL887。历史不足原partial风险及.4094748402105834接受门限保持；只校准历史充分分支，图像/回退修正版保留强对照。

|VAL各自工作点|接受|FA|FR|ED|CR|
|---|---:|---:|---:|---:|---:|
|均匀／原部分AUC|845|186|4|37|659|
|既有回退修正版|844|185|4|38|659|
|困难加权候选|848|188|3|35|660|

同848接受：候选/均匀部分AUC/原部分AUC/回退修正版FA188、FR3，simple195/10、score196/11、旧四维189/4。相同CR660：候选FA188，部分AUC186、回退修正版185、旧四维193、simple202、score201，没有增量。总体离线同CR659候选FA185，与旧回退修正版持平，但seq07只保留59/63=93.65%，不满足95%，不能选为新线上门限。离线比较保持GT评价身份。

候选各自工作点Real FA142/FR3/ED33/CR196，seq07 129/3/33/60，seq14 13/0/0/136，Sim 46/0/2/464。逐组95%通过，seq07同CR60相对回退修正版FA126→129。相对回退修正版仅4个VAL标志变化：seq07_00069/00150/00159三错误重新接受，seq14_00018一正确恢复；无原FA被新检出。低/高/无历史FA165/21/2，旧164/19/2。历史不足身份保护正确，没有重现旧回退门限问题。

TRAIN各自VAL门限均匀FA22/FR622/ED49/CR1865→候选31/361/40/2126；同2157接受/同CR2126原部分AUC也FA31，回退修正版24。18个困难TRAIN错误在两臂各自VAL门限下全部接受，没有成功检出它们。历史充分TRAIN AUROC .82748→.82983，VAL .78160→.76324，未形成所需排序收益。加权正常收敛，目标1.09018→1.03932，梯度3.52e−11；均匀控制还原原部分AUC。不同加权目标损失不可直接比较。31个失败条目含关联/重复指标，不是31个独立失败，主要同数/同CR无增量、Sim及连续正确FR保护失败。最长正确FR仍2帧，匹配simple/score1帧。

建议：停止本加权候选，不追加4倍/50%等常数、交互或预算，不自动TEST。保留原时序、部分AUC、回退决策研究信号。本结果不证明全部加权无效或唯一根因；目前这一同四维目标增量不能减少主要稳定FA，后续需有明确新增信息/机制依据。

工程：本地/服务器12项测试通过，3445行/3895项独立核验一致；解析梯度/Hessian、均匀还原、固定名单/同分/无VAL门限、权重归一化、模型先保存、GT/未来隔离、中心方向框score覆盖保护、回退门限/同数同CR/各组校准/连续段/门槛核验。服务器前端权重SHA验证，本地缺两份pth仅显式allow-absent核对缓存/回传。零检测推理/更新、零神经训练，没有改报告后重拟合。补充离线解释不参与选择。

GitHub main源码ab1faae已推送；服务器普通fetch仍被127.0.0.1:17890代理拒绝，SHA核验增量bundle仅导入本轮7个新增文件，原codex/oof-gradient-chain-v1提交16d572e，原暂存patch前后SHA d54105b5…保持。不冒充普通pull或两端同HEAD。mmrotljj、准确项目路径、CPU单线程，无GPU占用。

结果统一work_dirs/port_reliability_hard_error_pauc_v1/20261014_fixed_hard25x2_v1/result，含原报告/硬集合/两模型/校准/逐帧、双端审核、补充解释及收尾回执。报告SHA138d31ce14107e8614fceeed16eea83b53941b10346a7380bbb13f851a6ba050。只追加运行记录、保留草稿，不修改论文主要结果。

清理仅本次work_dirs/port_reliability_hard_error_pauc_v1_20261014_fixed_hard25x2_v1_analysis.tar.gz（75c9d69476e4f6918e6ca86a8cbb4b5169352b21aedce3a2312a2181b55cf26b）及work_dirs/port_reliability_hard_error_pauc_v1/source_ab1faae.bundle（c8f411874ed3494197b6c01abac19aeb29b724ac719dfcba5664fd58507f216e），按精确路径/非符号链接/SHA验证后清理，实际完成以双端cleanup_receipt为准；完整结果、权重和所有旧实验保持。


## 58. 原图边界支持：完整TRAIN前提检验（2026-10-14）

状态TRAIN_SCREEN_ONLY_NO_FIT。预设局部信号前提通过，但没有判断器拟合、门限校准、VAL/TEST或FA改善结果。不把“通过前提”写成方法被采用；正式M、simple与原四维/部分AUC/回退修正版全部保持。

区别于旧对边：旧候选取冻结256×9×9 ROI的通道均值/RMS。本项只用原图灰度、固定M中心/方向/长短边，在每组对边7条切向剖面上读取原像素±1梯度；从预测边长0.25—0.75位置的129个点中取最高梯度位置，同分取中位，记录相对预测边界的绝对/有符号log距离与跨剖面离散度。没有图像放大或额外图像分辨率、没有新检测/辅助网络。梯度峰可能来自背景、纹理或遮挡，不能视为真实目标边缘；左右侧保留原始剖面，不把平均偏移解释为修框。

完整TRAIN2558个原始输出全部读取并SHA核对原图，原冻结框与接受标志不变。完整TRAIN原研究参照仍FA18/FR580/ED53/CR1907，无MISSING。低时序偏差2372帧含56错误/2316正确，其中L_only33/S_only19/both4；剩余接受FA17/CR1852，另FR464/ED39。164个低偏差FA是旧VAL统计，本轮没有打开它们或用作训练名单。

|低时序偏差TRAIN|长轴边界距离|短轴边界距离|
|---|---:|---:|
|支持输出|2370/2372|2372/2372|
|全部错误/正确总体AUC，固定距离大为错|0.5377|0.4217|
|同视频、相近预测几何尺度配对AUC|0.5641|0.4480|
|原已接受FA/CR总体AUC|0.4448|0.4971|

固定前提：支持的低偏差TRAIN错误≥20；至少两个Real视频各有≥3错误/20正确且同视频预测尺度±10%匹配AUC≥.55；只报告预设正方向，不翻转或扫门限。两轴分别支持56个错误。长轴seq06/seq12匹配AUC .6243/.7372，通过；短轴seq12/seq13 .5631/.6904，通过。因此feasibility_passed=true，原门槛保留原身份，不能事后改成失败。

但是视频方向不一致：长轴seq13 .3933、seq05 .5169；短轴seq06 .3636、seq05 .4671。Sim只有1个低偏差错误，不支撑Sim泛化结论。小样本相关帧/配对数不视为独立样本，匹配不能排除所有混杂。

补充描述性诊断（不参与前提或选择）：只在原已接受集合做同视频/预测尺度匹配，L/S AUC .5612/.5641，分别1870/1872对，不是1870/1872个独立错误。按对应轴真实错误配对（L_only/both与正确，S_only/both与正确），L为37错误、AUC .5407；S为23错误、AUC .4723。不能据局部AUC解释为梯度峰准确定位目标真实边界，更不能称已经增加错误检出。

建议：本项建立了与旧ROI不同的可观测输入，且具备局部TRAIN线索；没有证明跨视频统一拒绝规则。若继续，最多提出一次继承原四维/部分AUC与冻结回退的轻量两轴证据增量对照，固定TRAIN预算与VAL保护合同，检验同CR和同接受数FA是否真的下降；不能直接“边界距离大就拒绝”、按视频选方向、扩展原图网络或针对164个VAL错误设规则。对应轴语义较弱必须披露，不能把前提通过包装为强收益依据。本轮不自动开展该训练，也不转入VAL/TEST。

工程：本地/服务器各12项测试通过，完整2558行/2576项独立核验；本地另用不导入提取核心的逐像素标量插值，按每个TRAIN视频首/中/末帧固定抽查18图/72侧剖面，全通过。GT只参与离线10%标签，不进入图像采样API；宽高交换/π周期/旋转、边界/平坦支持、缺框、输入保护和固定方向测试通过。完整前端未重新推理/权重现场核验，M来自已SHA固定审查缓存；不是新部署延迟测量或独立未见验证。

Git：main新增6文件提交31dc82a已推送；服务器原分支codex/oof-gradient-chain-v1导入提交bb63d68，旧暂存patch SHA前后d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e保持。服务器SSH第一次超时、第二次连接成功；普通fetch仍受127.0.0.1:17890代理拒绝，经SHA核验增量Git bundle导入。本轮mmrotljj/Python3.8.20、CPU单线程，不占GPU，不改变服务器代理、分支或其他作业。

原报告SHA：b77a28a7301f8692e16b34e0dafe05ab650f1bb2d3e60dea3268ed4c91052610。
结果统一work_dirs/port_reliability_pixel_border_v1/20261014_native_pixel_v1/result；只追加可靠性运行记录，不写论文主要结果。不覆盖报告或重新提取调参。

清理范围仅本轮回传包work_dirs/port_reliability_pixel_border_v1_20261014_native_pixel_v1_analysis.tar.gz（25200d1276b9d0b0318985f725d950872b5276824d90d986ebb457fd1cbe2b2c）和本轮source_31dc82a.bundle（2801a38992b97ab233bd5b80e778c24f1c5139a594e77b5fc5631ab33c320155）；分析记录同步后按精确路径/非链接/SHA清理，实际完成以两端cleanup_receipt为准。完整结果、源码、旧实验和用户文档草稿保持。


## 59. 原四维＋两轴原图边界距离：固定部分AUC对照（2026-10-14）

状态EVIDENCE_FAILED_STOP，不采用，不追加取样/特征/权重/预算搜索，不自动TEST。用户先授权仅本地代码修改，后明确授权服务器运行；已完成代码、审核、Git同步、真实CPU两臂拟合与完整VAL、回传核验。正式M/simple/policy、原四维/部分AUC/回退修正版均保持。

本轮与第58节不同：前提检验只发现局部TRAIN线索，本项实际检验这些线索是否提高尺寸判断。复用固定原图提取，唯一输入增量为L/S两项mismatch；不改变像素范围、最强梯度读取、趋势窗口、困难加权或使用新网络。四维控制与六维候选使用同2526个历史充分/两轴像素支持TRAIN（71真实错误），来自完整2558帧；原2528历史样本中2个正确输出像素不足。同一原四维标准化/截距，两项新量标准化仅由该TRAIN子集确定，std<1e-8取1。零斜率初始化，beta=.05/T=.1/L2=.1/Newton最多100均相同。四维原目标数值还原测试通过，控制按相同新支持子集重拟合，不冒充逐位还原原2528样本权重。

先保存TRAIN_prepare与两模型，再解析/提取完整已暴露VAL887。VAL有886输出，两轴像素支持883、活动分支871；历史不足或任一像素不足使用冻结原partial_auc4风险及.4094748402105834门限，缺框单列MISSING。两臂单独确定一个活动门限，完整VAL各域各视频正确保留≥95%。候选.41228377286377826，配对控制.40950798223177076。GT仅生成10%标签/离线评价，在线输入为当前原图灰度、当前M、分辨率、帧序号/reset，无GT/域/视频路由或未来。

|完整VAL各自工作点|接受|FA|FR|ED|CR|MISSING|
|---|---:|---:|---:|---:|---:|---:|
|配对四维控制|845|186|4|37|659|1|
|既有回退修正版（原实际标志）|844|185|4|38|659|1|
|六维像素候选|847|187|3|36|660|1|

同847接受：候选/配对4/原部分AUC/匹配回退融合均FA187、FR3，simple194/10、score196/12、旧四维189/5。同CR660：候选FA187、配对4/原部分AUC186、匹配回退融合185、simple202、score201、旧四维193。候选仍优于simple，但属于继承原时序/目标的既有研究收益，不能算新增原图证据增量。匹配pAUC/融合锁定新像素缺失回退；报告另保留原实际回退修正版及候选到其CR的离线比较，避免分支变化混淆。GT离线同CR门限不是在线校准门限。

|六维VAL|FA|FR|ED|CR|最长连续正确FR|
|---|---:|---:|---:|---:|---:|
|Real|141|3|34|196|2|
|seq07|128|3|34|60|2|
|seq14|13|0|0|136|0|
|Sim／seq10|46|0|2|464|0|

各组95%正确保留通过，不能写成覆盖门槛不满足。Sim同CR仍46错误接受，simple45、score44，未形成改善；seq07连续正确FR仍2，相应simple/score1。35项失败条目包含相关/重复组、控制、匹配检查：同CR_FA19、同CR连续FR6、同数FA3、同数连续FR3、原实际参照同CR_FA3、低偏差FA增量1，非35个独立失败。

逐帧新发现：较原实际回退修正版仅3个VAL标志改变，seq07_00069和00159两个错误重新接受，seq14_00018一个正确恢复；无原FA被新检出。前者低时序偏差、后者高偏差。按旧三类时序定义统一复算，低/高/无历史FA164/19/2→165/20/2；配对4为164/20/2。原报告的四类细分活动low163/high19/历史不足2/像素不足3，不能把163直接与旧164比较宣称改善，因两项低偏差FA及一项高偏差FA被列入像素不足。

完整TRAIN各自VAL门限：配对4 FA22/FR621/ED49/CR1866→六维22/584/49/1903。相同CR1903或1925接受，配对4同为FA22，没有新增辨识收益；匹配旧回退融合FA18。候选比原实际参照（18/580/53/1907）恢复26正确、误拒30正确、重新接受4错误，没有新检出原FA。低/高/无历史FA配对4与六维均21/1/0，旧回退17/1/0。

活动分支AUROC TRAIN .827366→.827544，VAL .781544→.773383。六维目标1.090178→1.014895，配对4最终1.015204；两模型均7次记录、驻点梯度约1e-15，独立梯度复算通过。这是正常收敛但无所需FA增量，不是工程失败或预算不足。六维新增标准化斜率L+.006831、S−.013702；S方向与“距离大表示错误”直觉相反，是固定TRAIN线性目标的结果，未根据VAL翻转/搜索方向。小系数、下降的loss不证明新增信息无用或实际边界被准确定位。

建议：封存这一“两轴最强梯度距离＋线性pAUC”组合，保留原时序/pAUC/回退成果。第58节的局部AUC只提供实验前提，本轮证明它未转化为跨视频判断收益；原图最强梯度不能被当成可信目标边界。目前尚未验证峰来自背景/纹理/遮挡的具体比例，不能将其宣布为唯一根因，也不能宣布整个时序/原图可靠性已达理论上限。暂不换取样范围、符号、离散度或互动项凑新版本；下一项需先说明如何获得可信目标边界或其他有区别的信息，以及相对旧图像/ROI路线的实质增量，当前未指定有效新结构。

工程：本地和服务器各18项测试通过，含四维目标还原、六维梯度/Hessian有限差分、恒定新增量控制还原、合成完整执行器保存顺序、原回退锁定、同数/同CR/逐组95、GT/未来隔离、原图尺寸与宽高交换/π周期、中心方向原框score保护。完整3445行/3895项双端独立复核通过；服务器重新提取全部887VAL原图并核对证据，本地另以不导入核心的标量像素插值固定抽查各VAL视频首/中/末有效输出，9图/36侧剖面一致。服务器两份正式前端权重SHA实际核验；本地缺两份pth仅显式allow-absent审核缓存/回传，不冒充完整前端现场核验。零检测推理/更新、零神经训练、零TEST/深度评价。

Git：main新增7文件4df2115已推送；服务器原codex/oof-gradient-chain-v1分支导入12dfae5。旧暂存patch SHA前后d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e保持。普通fetch仍受127.0.0.1:17890代理拒绝，经SHA核验单提交Git bundle导入，不冒充正常pull或两端同HEAD。mmrotljj/Python3.8.20、CPU单线程，没有占GPU或修改服务器代理/分支/其他作业。

报告SHA：e4e0eaa23dd1ffeb04b5d7000d6ef834b81e8bb21538edc716139758ee7dec02。完整结果统一work_dirs/port_reliability_pixel_pauc_v1/20261014_paired_pixel_v1/result。只追加现有运行记录，保留用户草稿，不写论文主要结果，不改原报告/模型/门限后重选。

收尾只清理本次回传包work_dirs/port_reliability_pixel_pauc_v1_20261014_paired_pixel_v1_analysis.tar.gz（e3d64c0c34b068d18244777f87f2dec86cb8f38d2b071ae72934e2161b4cfb92）及source_4df2115.bundle（c715446e434c1bc6b18ed14fe2541b62a9df5f29eaa8217d948e995159d28dfe）。分析/记录同步后按准确路径、非链接及SHA执行，两端实际删除以cleanup_receipt为准；完整结果、逐帧gzip、代码和旧实验保留。


## 60. 冻结时序／部分AUC／回退版本：完整TEST补评（2026-10-14）

# 冻结时序／部分AUC／回退版本：VAL与已暴露TEST补评

所有版本在打开本次TEST前固定，未训练、校准或按TEST挑选版本。保留原VAL失败事实，不自动采用。最初时序硬拒绝与图像融合用于来源对照。

|数据|方法|接受|FA|FR|ED|CR|接受后错误|正确保留|全帧正确覆盖|最长正确FR|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|VAL|原图像融合|847|188|4|35|659|22.20%|99.40%|74.30%|2|
|VAL|修正回退融合|844|185|4|38|659|21.92%|99.40%|74.30%|2|
|VAL|原四维时序融合|846|188|5|35|658|22.22%|99.25%|74.18%|2|
|VAL|部分AUC|845|186|4|37|659|22.01%|99.40%|74.30%|2|
|VAL|正式simple|843|190|10|33|653|22.54%|98.49%|73.62%|1|
|VAL|最初时序硬拒绝|785|155|33|68|630|19.75%|95.02%|71.03%|2|
|TEST|原图像融合|1390|313|12|29|1077|22.52%|98.90%|74.79%|2|
|TEST|修正回退融合|1383|308|14|34|1075|22.27%|98.71%|74.65%|2|
|TEST|原四维时序融合|1383|310|16|32|1073|22.42%|98.53%|74.51%|2|
|TEST|部分AUC|1383|308|14|34|1075|22.27%|98.71%|74.65%|2|
|TEST|正式simple|1330|275|34|67|1055|20.68%|96.88%|73.26%|3|
|TEST|最初时序硬拒绝|1250|244|83|98|1006|19.52%|92.38%|69.86%|4|

固定门限逐视频TEST：

|视频|方法|FA|FR|ED|CR|正确保留|接受覆盖|最长正确FR|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|sequence:real_seq03|原图像融合|83|3|19|88|96.70%|85.50%|1|
|sequence:real_seq03|修正回退融合|80|4|22|87|95.60%|83.50%|2|
|sequence:real_seq03|原四维时序融合|79|6|23|85|93.41%|82.00%|1|
|sequence:real_seq03|部分AUC|80|4|22|87|95.60%|83.50%|2|
|sequence:real_seq03|正式simple|59|0|43|91|100.00%|75.00%|0|
|sequence:real_seq03|最初时序硬拒绝|39|19|63|72|79.12%|55.50%|2|
|sequence:real_seq04|原图像融合|208|9|10|439|97.99%|96.86%|2|
|sequence:real_seq04|修正回退融合|206|10|12|438|97.77%|96.41%|2|
|sequence:real_seq04|原四维时序融合|210|10|8|438|97.77%|97.01%|2|
|sequence:real_seq04|部分AUC|206|10|12|438|97.77%|96.41%|2|
|sequence:real_seq04|正式simple|194|34|24|414|92.41%|91.02%|3|
|sequence:real_seq04|最初时序硬拒绝|187|48|31|400|89.29%|87.87%|4|
|sequence:sim_seq09|原图像融合|22|0|0|550|100.00%|100.00%|0|
|sequence:sim_seq09|修正回退融合|22|0|0|550|100.00%|100.00%|0|
|sequence:sim_seq09|原四维时序融合|21|0|1|550|100.00%|99.83%|0|
|sequence:sim_seq09|部分AUC|22|0|0|550|100.00%|100.00%|0|
|sequence:sim_seq09|正式simple|22|0|0|550|100.00%|100.00%|0|
|sequence:sim_seq09|最初时序硬拒绝|18|16|4|534|97.09%|96.50%|2|

TEST同接受数和同正确保留：仅离线使用GT，不回写线上门限。回退版本使用risk/该帧冻结分支门限作为排序margin，1为其实际交付点；不是校准概率。

|候选|候选接受／CR|候选FA|simple同数FA／FR|score同数FA／FR|simple同CR FA|score同CR FA|四维同CR FA|部分AUC同CR FA|
|---|---|---:|---|---|---:|---:|---:|---:|
|原四维时序融合|1383／1073|310|301／7|311／17|285|311|309|304|
|部分AUC|1383／1075|308|301／7|311／17|289|312|315|308|
|修正回退融合|1383／1075|308|301／7|311／17|289|312|315|308|

分域及逐视频完整同数、同CR、同分界、AUROC/pAUC、连续段、低／高／历史不足FA与中心覆盖均保存于report.json。


本轮结论与新发现：

1. VAL局部增量未延续为TEST优于simple：部分AUC在VAL同845接受FA186、simple192；TEST同1383接受FA308、simple301，方向相反。不是仅门限更宽导致：同CR1075，部分AUC/回退FA308、simple289、score312。GT匹配点仅离线，不能据此重新设置TEST门限。
2. 部分AUC相对原四维仍有小幅增量：TEST同1383接受FA310→308、FR16→14；同CR1075，原四维FA315→部分AUC308。回退修正与部分AUC的实际TEST接受集合逐帧完全一致；不能把两者数值相同直接当集合相同，本轮已逐帧核实。
3. 最初时序硬拒绝TEST FA244/FR83/ED98/CR1006；同1250接受simpleFA239/FR78、score276/115。只比score好，未比simple好；正确保留92.38%，最长正确FR4，未达到原95%保护。VAL同785接受原候选155/33、simple170/48，体现显著场景差异。
4. TEST分域/视频：部分AUCReal FA286/FR14/ED34/CR525，Sim22/0/0/550。seq03 80/4/22/87，同167接受simple76、同CR87simple47；seq04 206/10/12/438，同644接受simple206、score205，同CR438simple205、score203。seq03正式simple为59/0/43/91，seq04为194/34/24/414。Sim全接受22个错误，无新检出。seq03主要被改差，seq04没有稳定优势。
5. 部分AUC完整输出AUROC VAL .7873→TEST .7622，simple .7856→.7666；seq04部分AUC .5409、simple .5434、score .6339。整体AUC受域间/视频间差异影响，不能代替视频内高正确保留FA指标。低时序偏差TEST FA281/308=91.23%，支持“稳定错误难辨识”的限制，不能宣布唯一根因或时序理论上限。
6. 高接受数量存在下界：TEST1431输出中1089尺寸正确，接受1383时即使完美排序仍至少FA294。本轮308与这一理想下界差14，说明必须同时关注正确保留/使用点，不能期望保持当前接受数时FA大幅归零。这个算术下界不是模型已达到最优的证明。
7. 中心不受筛选：TEST输出1431/1440=99.375%，中心命中1430/1431=99.930%，全帧正确中心覆盖1430/1440=99.306%；VAL874/886输出命中、886/887输出覆盖、874/887全帧正确中心保持。缺失9个TEST与1个VAL均MISSING，不参加FA/FR/ED/CR。

建议：保留正式M/simple，不按TEST重选权重、门限或新版本。研究结果应同时写明“部分AUC比四维有小幅收益”与“没有跨场景优于simple”；原VAL失败事实保持。后续不自动追加时序窗口、趋势、目标常数或融合参数搜索。本轮把TEST补全为冻结报告，结束仅凭VAL宣称泛化收益的缺口；新候选需先有能区分稳定尺寸错误的新证据和TRAIN/VAL固定合同。现阶段不指定已证明有效的新结构。

工程与来源：本机/服务器各9项检查通过，2327帧/3181项双端核验一致；CPU，零拟合/校准/检测推理/更新，未改变权重/policy/框/score/中心/方向/输出身份。服务器两份前端SHA实际验证；本机缺两份pth仅显式allow-absent复核，不冒充现场权重核验。审核原版曾因独立浮点计算落在精确VAL门限另一侧，以及内存整数JSON键与磁盘字符串键直接比较而失败；新审核器只修复审核，将独立风险1e-10内的门限同分按冻结交付风险处理并统一JSON键，报告/评分/模型/门限从未重跑或改写。最终审核源码SHA c3ff942aca68a0e55af2cc87c27632ea62f30e2073a6d04d6736e0eec5fbb8cc，原失败日志保留。评分源main5cf9342，审核修复最终main005cef9；服务器原分支通过核验Git bundle及精确文件导入，非普通pull、非两端同HEAD，旧暂存patch保持。

原报告SHA a5c29f7aa3cad07900f414d425ad6babe304bd58f834292b45874673c0a1e7c9。完整结果保存本目录；只删除本轮传输tar和source bundle，逐帧gzip与所有旧实验保留。TEST已多次暴露，不能当作未触碰独立验证，不参与后续选择。


## 61. 完整TRAIN：simple剩余错误与已拒绝错误的监督支持（2026-10-14）

# 完整TRAIN剩余尺寸错误监督支持核对

本项是离线材料核对，没有训练判断器或改善FA的性能结果。正式M/simple与三分量输出保持。

|TRAIN视频|FA|FR|ED|CR|
|---|---:|---:|---:|---:|
|all|13|1037|58|1450|
|sequence:real_seq01|1|0|0|338|
|sequence:real_seq05|0|516|44|0|
|sequence:real_seq06|0|453|13|0|
|sequence:real_seq12|4|16|1|120|
|sequence:real_seq13|7|52|0|245|
|sequence:sim_seq08|1|0|0|747|

本轮新事实：

- 完整2558个TRAIN输出中71个尺寸错误：simple接受13、拒绝58；正确输出2487：接受1450、拒绝1037。冻结门限下TRAIN正确保留1450/2487=58.30%，这不是模型训练准确率，也不能用VAL的覆盖约束代替TRAIN交付表现。
- seq05和seq06全部1026帧的尺寸标志均拒绝，其中正确969、错误57。它们占58个ED的98.28%，以及1037个FR的93.44%。seq01全部339帧接受；Sim全部748帧接受。TRAIN接受决策明显受视频分布影响，但不证明按视频路由、标注或某个特征为唯一根因。
- 13个FA来自seq01/12/13/Sim，各为1/4/7/1；属于12个最大连续错误段，仅seq13的00004和00005共享一段。这些段不能被当成已证明独立采集事件。
- 全71错误共51个最大连续段。13个FA中12个低时序偏差、1个高偏差；长边超限8、短边超限5，无双边超限。58个ED中长边32、短边18、双边8；FA与ED在全TRAIN上均有相同单边偏大/偏小类型。
- 所有13个FA都有其他视频上的同轴同符号ED支持，但同视频同签名ED均为0。严格匹配为零首先发生在视频与类型条件，不是后续±.1尺寸带、3帧排除或片段限制造成。保留原标准，不扩大匹配条件去制造通过。
- 跨视频相同误差类型不等于图像外观/特征/真实机制相同，不能将这58个ED全部宣布可替代13个FA，也不能宣布它们完全无用。

固定材料前提未通过：剩余FA13；唯一严格匹配ED0；扩展支持仍13，小于预设20，达到每视频>=5错误且>=2错误段的Real视频仅seq13，少于2。20/5/2是预先固定的保守描述性计数条件，没有统计保证，失败不证明不存在有效模型。

本项没有取得特征可分性的结论：严格ED与CR联合可比FA为0，static3/native256两块都不存在可计算的成对ED-CR距离。因此“ED更近0”不是AUC0、不是特征无效，更不是已经证明网络能力上限。原生特征仍完整读取、按全TRAIN标准化并核对配对。跨视频距离不自动承担语义可复用证据。

建议：不只用13个FA训练二级头，不继续把旧评分交集或时序规则包装为新机制，也不据此要求补相似视频。下一项若继续，应使用完整TRAIN71个真实错误，并先提出能区分视频分布依赖与真正尺寸证据的固定检验；可考虑对现有轻量判断器做TRAIN内部按视频留出评价，验证在其他视频拟合的规则能否识别留出视频错误。这不是重新训练辅助检测器，不改变TRAIN/VAL/TEST身份；当前未实现或运行该步骤，尚无证据支持特定新结构。检测前端及旧PCA已经见过全部TRAIN，即使未来仅判断器按视频留出，也不能称完整检测链的未见视频验证。若跨视频证据不足，应停止增加判断器，而不是追加损失权重或容量。

范围与工程：仅TRAIN，零拟合/门限更新/检测推理/神经训练，无新增VAL/TEST读取或选型。GT只用于误差类型及离线匹配，不进入在线API。中心、方向、原框、score、数量及输出身份保持；缺框单列MISSING，本TRAIN无缺框。

本地/服务器各10项测试通过，2558帧/2610项双端独立核验。测试覆盖四状态/10%边界、长短边交换、错误片段不因接受标志切换而拆分、漏检/断帧/采集边界、严格匹配及GT角色、缓存配对、特征有限与输入不变。服务器两份正式前端权重SHA核验；本地缺两份权重仅显式allow-absent复核。保留原图像PCA与缓存来源，未重新提取或重拟合PCA。

命名修复：初次提交e57b2ac与旧TRAIN支持检查的测试/来源清单重名；6946420恢复两份旧文件原始字节并将本轮命名为remaining_error_support_v1。最终相对005cef9只有本轮7个新文件；服务器只导入这7个新路径，旧文件SHA及既有index前后保持，没有导入中间覆盖状态。GitHub main6946420已推送；服务器仍原分支，由核验增量Git bundle同步，普通fetch受127.0.0.1:17890代理拒绝，不冒充正常pull或两端同HEAD。

本轮前提/原报告保持，补充匹配阶段与特征分布只作描述，没有根据结果改变筛查、再训练或选择候选。

原报告SHA：812ce7aca97ccb168ca0d89427b6ea904ecb6114b9645b251372dde8fd3e7d3d
结果统一保存在本目录；只清理本次传输tar和Git bundle，完整报告、片段、匹配、审核与旧实验保留。


## 62. 完整TRAIN判断器训练内与按视频留出迁移检验（2026-10-14）

# 完整TRAIN判断器视频迁移检验

结论：本轮定位到“跨视频工作点迁移失配”和“高正确保留区域错误排序不足”同时存在。没有发现判断器求解未收敛；没有有效新版本，正式M/simple保持。本轮是诊断，不是可靠性提升的性能结果。

## 固定范围

2558个TRAIN输出/71个真实尺寸错误/2487个正确。复用完整缓存、原三维simple和原始四维时序偏差；完整TRAIN拟合作训练内参考，六次按视频留出。留出视频不进入判断器拟合、均值/方差、类别权重、门限。保持原balanced Newton/L2=.1/max100/gradient1e-8。每个上下文沿用原full static fallback、supported static和supported fusion三次拟合，共7×3=21次轻量优化；六视频两臂共12个留出评价。full static模型精确复现正式simple参数；支持不足使用本轮只由拟合视频训练的三维回退。

门限：每种评分按FIT各视频正确输出95%分位点取最大值，形成一个全局诊断门限，再原样应用于留出视频。这不是原policy的VAL全帧接受95%门限；两者不混称。正确95%排序点、同接受数/同CR只作离线GT评价，不作为交付门限。full_simple已在全部TRAIN训练，只能作历史参考，不是公平留出学习器。score是检测score；新的三/四维才是判断器留出对照。没有 pooled cross-fold AUC，因为不同fold分数尺度不同；合并FA等计数只描述六个内部诊断。

## 逐视频高正确保留区域

以下各列都在同一视频内保留 ceil(.95×正确输出数)，本轮无边界同分额外CR，因此可直接比较FA。这些门限只供离线比较。

|视频|训练内三维FA|留出三维FA|训练内四维FA|留出四维FA|score FA|
|---|---:|---:|---:|---:|---:|
|real_seq01|1|1|1|1|1|
|real_seq05|32|29|35|35|42|
|real_seq06|10|11|10|10|11|
|real_seq12|5|5|4|4|5|
|real_seq13|7|7|7|7|7|
|sim_seq08|1|1|1|1|1|

合计各视频同正确保留2367/2487=95.18%：训练内三维FA56，留出三维54；训练内四维58，留出四维58；score67。四维在高保留区域整体不优于三维，不能只凭完整AUC提升采用。不能把内部合计当新的固定全局policy或独立泛化结果。

## 真正迁移的诊断门限

|留出视频|方法|FA|FR|ED|CR|正确保留|最长正确FR|
|---|---|---:|---:|---:|---:|---:|---:|
|seq05|三维|3|500|41|16|3.10%|79|
|seq05|四维|12|257|32|259|50.19%|9|
|seq06|三维/四维|13|0|0|453|100%|0|

其他seq01/12/13/Sim两臂均全接受，对应错误1/5/7/1全部接受。合并留出三维FA30/FR500/ED41/CR1987，四维39/257/32/2230；这两者接受数不同，不作FA提升结论。原冻结policy的TRAIN13/1037/58/1450保持原身份。

seq05三维AUC训练内.6652→留出.6721（略升），却因其他视频确定的门限只保留16个正确框。它的离线同正确95%FA32→29，说明绝非所有下降都能归因留出后排序变坏，也不是完全没有排序信息。四维seq05 AUC.6751→.6558，同正确95%FA保持35，未优于三维29。seq06三维AUC.5828→.4328、四维.7040→.6274，表明部分视频确有排序迁移损失，不能把全部问题仅归因门限。

## 与旧结论衔接

完整TRAIN pooled三维AUC.8018、四维.8314，但三维训练内seq12/13 AUC仅.4765/.3506；总体分数差异不能代替视频内辨识。四维提升seq12/13排序（留出.7132/.7133），但同正确95%仅seq12多检出1个错误，seq13仍接受7/7错误。单错误视频seq01/Sim不能从AUC或一帧计数宣称稳定收益。

本轮证实的两层限制：
1. 绝对分数/工作点对视频分布敏感，既有阈值可把一个视频大量正确输出全部或几乎全部拒绝。
2. 现有三/四维在高正确保留区域仍接受多数错误，时序改善整个排序并不等于错误落入可拒绝尾部。逐视频95%点，留出三维仍接受54/71错误，四维58/71。

不证明：类别平衡是唯一根因、标签错误、网络能力上限或图像原生特征无效；本轮没有类别权重反事实/新图像特征实验。71个错误集中且视频相关，检测前端已经见过这些TRAIN；不能称整个检测链的未见视频测试或风险保证。

建议：保留正式M/simple，当前四维只保留既有增量事实。停止在相同三/四维上继续改全局门限或增加训练步数；单纯分数单调校准不能改变同CR错误排序。也不自动重开既有partialAUC、within-video rank、加权、原图/ROI参考或辅助模型。若继续优化FA，新设计必须增加能识别稳定尺寸错误的有效证据，或明确给出与旧失败目标不同的训练机制与固定对照；本轮尚不足以支持某个有效结构，先不启动额外训练。已有可靠性限制可以准确进入论文讨论，不能因TRAIN FA低宣称训练成功。

## 工程与来源

本地和服务器各10项测试通过。独立审核检查2558帧、5116条评分、5469项：视频排除、标准化/类权重、21个优化目标的收敛梯度、因果历史与回退、门限来源、逐域逐视频FA/FR/ED/CR、同分/同数/同CR、配对AUC和连续正确FR。审核未重新优化模型。server两份正式前端SHA实际核验；本地缺两份pth显式allow-absent，不冒充权重现场检查。

正式框、score、输出/缺失身份、中心、方向、原policy均保持。GT仅离线。本輪CPU，无GPU/检测推理/神经网络训练或新增VAL/TEST评价。旧native缓存由共用loader核验，未作为特征拟合/重PCA。

本机main c128ffa已推送。服务器普通fetch因127.0.0.1:17890代理拒绝失败，使用SHA核验增量Git bundle精确导入7个新文件；第一次按HEAD取bundle引用失败，改为实际main引用后成功，失败时未改项目文件。服务器保留codex/oof-gradient-chain-v1，提交b67f0a2；既有暂存patch SHA d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e前后保持。不是两端同HEAD/普通pull。

报告SHA：98b74d2220a8289256feb664e8d8817779759e523e6260acb1dc5db2418bab07。
结果集中本目录。记录/双端核验后，只删除本次transport tar和source bundle；完整结果、原权重/缓存/历史代码保留。未修改论文主要结果文件。


## 63. 三维尺寸判断器逐视频FR约束目标有限对照（2026-10-14）

结论：工程与约束求解审核通过，候选性能未通过，不采用，停止于TRAIN，不进入VAL/TEST。保留正式B24＋σ1.5/epoch03与原simple/policy。约束保护了正确输出，却在零门限下退化为全接受；离线重设工作点后排序也变差，因此不是只差门限。只证实本次三维线性模型、固定系数、保守代理约束组合无效，不证明可靠性无法优化或当前特征已达到理论上限。

### 目的、合同与实现

本轮检验训练目标是否能同时减少错误接受并避免整段正确视频被拒绝。固定完整TRAIN2558输出（2487正确、71错误）、原score logit/相对几何尺寸log/长宽比log三个特征、同线性容量和拟合视频标准化。正式M输出和三个使用标志、框、score、数量/缺失身份、中心和方向均保持；只计算离线尺寸候选分数，不导出在线新policy，GT仅监督/评价。

控制：原balanced logistic/Newton/L2=.1/max100/梯度1e-8。完整TRAIN控制精确复现formal simple均值、方差与权重；门限为拟合各视频正确95%分位最大值，并非原VAL全帧95%policy。候选：风险logit s=wᵀx+b，最小化错误样本mean softplus(-s)+.1/2||w||²（截距不惩罚），每个拟合视频正确样本mean softplus(s)/log(2)≤.05；s<0接受，s≥0拒绝。可行性只上界拟合视频经验FR，不保证未来视频风险。借鉴约束分类问题的项目适配，不是完整NP算法复现。

NumPy float64 CPU可行内点Newton，固定mu=.1,.01,.001,.0001,.00001,.000001,.0000001，最多100步/阶段、60回溯、Armijo=.0001，梯度≤2e-7；独立KKT站立残差≤5e-6、对偶间隙≤1e-6、约束可行。中性可行初始化为w=0/b=log(expm1(.025 log2))。无随机性、重启、系数搜索或新增图像/ROI证据；无GPU/检测推理/辅助模型。原native缓存仅由继承loader核验。

完整TRAIN两臂训练内参考，六次TRAIN内部按视频排除（每次两臂，共14次拟合）；这只检验判断器迁移，检测前端已见过这些TRAIN。没有新增外部fit/probe角色，没有混合不同fold分数计算全局AUC。历史VAL/TEST已反复暴露，未读取/拟合/选门限。

预定门槛：每个留出视频实际正确保留≥95%；各视频同正确95%合计FA严格小于同三维控制54；逐视频同正确95%FA不退化；实际交付同接受数/同CR相对控制、历史simple、score均不退化，连续正确FR按同CR保护。失败停止，无结果后调整容忍值。可行、loss或总体AUC改善均不能独立通过。

### 结果：真实迁移工作点

|方法|接受数|FA|FR|ED|CR|正确保留|最长正确FR|
|---|---:|---:|---:|---:|---:|---:|---:|
|三维控制留出合计|2017|30|500|41|1987|79.90%|79|
|约束候选留出合计|2558|71|0|0|2487|100%|0|

六视频候选均全接受。接受数量不同，上表不能拿FA30→71单独评价辨识；FA71/FR0说明本次零门限没有检出任何尺寸错误。完整TRAIN训练内参考也全接受，候选FA71/FR0/ED0/CR2487，控制FA59/FR25/ED12/CR2462（控制接受2521）。这是新合同诊断，不覆盖原冻结policy的TRAIN计数13/1037/58/1450。

### 排除门限因素：各视频相同正确保留

离线各视频保留ceil(.95×正确数)，两臂边界无额外CR同分，合计CR2367/FR120；这些GT门限仅评价，不进入在线。

|留出视频|控制FA|候选FA|控制ED|候选ED|
|---|---:|---:|---:|---:|
|real_seq01|1|1|0|0|
|real_seq05|29|41|15|3|
|real_seq06|11|13|2|0|
|real_seq12|5|5|0|0|
|real_seq13|7|7|0|0|
|sim_seq08|1|0|0|1|
|合计|54|67|17|4|

候选FA增加13、错误检出减少13，主要seq05(+12)、seq06(+2)，Sim仅1帧增益不足抵消。Sim只有1错误，不能据此声称稳定Sim收益。留出seq05 AUC .6721→.5074，seq06 .4328→.3880。完整TRAIN训练内pooled AUC .8018→.6861；总体pooled正确95%点FA48→64，和上表逐视频合计不是同一个工作点。

候选相同实际CR2487下，控制与历史simple分别仍可排除部分错误，故没有同CR收益。预定6项失败理由均由独立审核复核，不把数量当与旧实验可比的严重性评分。失败后未开展VAL/TEST、未放宽.05、未重选门限或权重。

### 新发现、推断与边界

事实：每个拟合视频代理FR约束实际满足；所有14个拟合目标均通过独立数值审核。候选同正确95%排序退化且零门限全接受。没有证据表明本轮因损失方向/求解不收敛而失败。

机制推断：softplus(s)/log2≤.05比允许5%硬FR保守很多，它也惩罚所有仍被正确接受的正确框，倾向把整个风险分数推负；相同低维线性输入在约束下没有形成足够有效的错误尾部。本轮没有独立分离代理约束、正则系数和特征重叠各自贡献，不能宣布唯一根因，也不宣称任何FR约束方案必然全接受。

建议：封存本次保守代理目标，保留原simple及历史时序增量事实；不自动放宽正确保护、调L2/代理门限或重开辅助训练。下一设计需要与这次代理目标实质不同并说明新增错误辨识依据。当前结果不支持以“FR归零”作为可靠性或下游深度的改进。未修改深度公式/几何窗口/论文主要结果。

### 验证、同步与清理

本地与服务器各12项测试通过，涵盖解析一/二阶梯度有限差分、损失方向/截距、可行证书、无信息全接受反例、留出GT隔离、标准化、零门限/缺失、单类别失败以及篡改对偶乘子/均值的审核检出。两端独立审核2558帧、36158项，包含14次目标证书、逐视频约束、四状态/连续FR、同数同CR/同分、AUC、固定继续门槛，审核不重新拟合。服务器两份正式前端SHA现场核验；本地缺两份pth使用明确allow-absent，未冒充权重现场校验。

源码main a00eefe已推送。服务器普通fetch仍因127.0.0.1:17890代理拒绝失败，SHA核验增量Git bundle精确导入7个新增文件。服务器保留codex/oof-gradient-chain-v1，由b67f0a2→bcce610，原有暂存patch SHA d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e不变；不是两端同HEAD或普通pull。报告SHA 11222fe2f7be88bd14363f512050cd7ad737ff743d03ef30780e3e0d12261029。

结果目录：work_dirs/port_reliability_fr_constraint_v1/20261014_fr_constraint_v1/result。CPU，不占GPU。权重、原缓存和旧结果保留。回传SHA与双端独立审核一致、记录同步后，只清理本轮transport tar和增量source bundle，清理凭据存于该结果目录；未删除其他版本文件。

已完成的服务器指令（无需补跑）：
```bash
cd /media/omnisky/personal_files/ljj/symEOOD
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
bash tools/run_port_reliability_fr_constraint_v1.sh 20261014_fr_constraint_v1
```


## 64. 冻结simple/TRAIN保护工作点与约束候选的完整三数据补评（2026-10-14）

用户明确要求整理现有结果并补齐VAL/TEST。本轮只用冻结完整TRAIN模型、TRAIN保护门限和旧缓存CPU评分；覆盖上一轮失败候选的补评授权，不撤回第63节TRAIN失败，不自动采用。8项测试通过，4885帧5410项独立检查通过，无拟合/重新检测/GPU/服务器调用、门限或模型修改。缺失帧独立，不进入FA/FR四状态。原框/score/中心/方向身份保持。

原simple保护风险门限0.7335182395480431；候选logit<0，零也是门限。TRAIN保护版本FA59/FR25/ED12/CR2462，候选71/0/0/2487；VAL两者223/0/0/663（缺失1），TEST两者342/0/0/1089（缺失9），在VAL/TEST均全接受，与raw集合相同。原simple分别VAL190/10/33/653、TEST275/34/67/1055保持。两者不构成新有效基线，不从TEST挑门限。

完整总体/分域/逐视频、同接受数和同CR、中心输出覆盖与输出帧命中已写本次report/CSV；结果资料文档第13节集中整理。结果路径work_dirs/port_reliability_frozen_workpoints_v1/20261014_frozen_summary_v2/result，report SHA0d66076bf8574459da520ba8e4169744367603e8aff71ea4c04c130df3ea96bd。首次v1在评分前原生sim TRAIN split=train_sim校验失败，保留失败标记；校验修正后v2完成，原数据标签保持。没有新回传压缩包或逐帧缓存副本，没有清理其他结果。源代码未提交/未推送，本轮记录仅本地。实际论文稿保持。


## 65. 实际FR计数约束的三维线性有限对照：固定预算未决（2026-10-14）

结论：代码/逐帧审核通过，本轮没有形成可采用版本，停止于TRAIN，不进入VAL/TEST。与上一轮保守softplus约束不同，本轮完整TRAIN确实找到一个FR0、ED7的线性可行解，但七个求解均超时未证最优，其中两折没有求解器候选。不能据此宣称真实计数目标无效、三个特征已到能力上限或NP方法不适配。正式M＋原simple＋原门限保持主比较身份。

### 目的、固定合同和在线边界

本轮回答：在各拟合视频实际正确保留至少95%的条件下，现有score logit、相对几何尺寸log、长宽比log是否可支持更少FA的线性边界。完整TRAIN2558输出：71尺寸错误、2487正确；仅GT真实双边≤10%标签离线监督。复用上轮完整TRAIN和六次按视频留出，检测前端已见过全部TRAIN，不能称整个检测链未见视频测试；不创建新外部数据角色。

风险s=wᵀ标准化x+b，s<0接受，s≥0拒绝。每行整数变量z∈{0,1}与真实拒绝符号一致；最小化FA=Nwrong−sum(z_wrong)，各拟合视频sum(z_correct)≤floor(Ncorrect/20)。同时约束s−M_i z≤−ε、s−M_i z≥ε−M_i；ε=.0001，四个系数（包括截距）均在[-10,10]，M_i=10sum(abs([1,x_standardized]))+ε。所有拟合输出需位于符号间隔外，范围受数值参数限制，不是无限制线性模型上限。标准化只读本次拟合视频；没有按在线视频/GT选择路由。

SciPy1.10.1 milp/HiGHS，固定每次60秒、最多100000节点、相对gap0、presolve=True；不重启搜索、不调预算/范围/间隔。完整TRAIN+六留出共7次求解。控制参数、均值/方差和FIT各视频正确95%分位最大门限，从上一轮独立核验的report原样复用，control_refits=0；源report SHA11222fe2f7be88bd14363f512050cd7ad737ff743d03ef30780e3e0d12261029。完整控制仍精确对应formal simple，门限属于诊断，保留原policy门限。

状态分清：solver可行解≠solver已证最优；超时的可行解保留FA上下界/gap；没有solver可行解时仅输出明确标记的全接受参考，不称新训练候选。记录的dual bound是求解器报告，独立审核验证实际计数、符号和界的自洽，不重建分支定界证明。可行不保证未来视频正确保留或任何安全风险，仍是NP思想启发的项目适配。

预定门槛：完整TRAIN在同CR下严格减少FA；各留出视频实际正确保留≥95%；各视频95%正确保留目标合计FA严格低于控制54；逐视频同接受数/同CR FA及同CR连续正确FR受保护；任一折没有solver候选即失败。未通过停止于TRAIN。不存在通过日志或放宽保护后自动进入VAL/TEST的操作。

### 完整TRAIN训练内可行解

|方法/工作点|FA|FR|ED|CR|接受数|
|---|---:|---:|---:|---:|---:|
|原simple＋原policy|13|1037|58|1450|1463|
|原simple＋TRAIN保护诊断门限|59|25|12|2462|2521|
|本轮完整TRAIN可行解|64|0|7|2487|2551|
|同CR2487的simple离线点|70|0|1|2487|2557|
|同接受2551的simple离线点|65|1|6|2486|2551|
|全接受参考|71|0|0|2487|2558|

本轮确有局部训练内信息：相同CR2487时，比simple少6个FA；相同接受2551时，少1个FA、少1个FR。这不应被抹去。7个错误检出全部位于real_seq05：FA44→37、CR516保持。其他视频没有实际错误检出。训练内总AUC .7772，原simple .8018；局部高保留收益不等于总体排序更强。

但本轮目标是在允许每视频最多5%正确误拒时最小化FA。已核对原控制保护点是本轮数值范围内的可行解（仅正缩放/在相邻训练分数间表达同一接受集合，无重新拟合），FA59，低于本轮solver64。因此当前可行解连已知可行控制的主目标都未超过；允许FR计数不同，不把这一比较称同CR性能比较。完整TRAIN solver报告FA下界0/当前解上界64；另有已核验控制可行证据收紧已知可达上界至59，但未改变solver/model/report。明确不是FA64的最优结论。

### 求解预算与逐视频迁移

|上下文|solver候选|拟合FA|拟合FR|已核验控制可行FA|状态|
|---|---|---:|---:|---:|---|
|full_TRAIN|有|64|0|59|60秒超时，未证最优|
|real_seq01|有|67|0|59|60秒超时，未证最优|
|real_seq05|无，仅全接受参考|27|0|24|60秒超时，未证最优|
|real_seq06|有|46|1|43|60秒超时，未证最优|
|real_seq12|有|56|0|54|60秒超时，未证最优|
|real_seq13|无，仅全接受参考|64|0|53|60秒超时，未证最优|
|sim_seq08|有|68|0|59|60秒超时，未证最优|

七次均未证明最优；seq05/seq13两折没有solver候选。实际留出六个视频均全接受：合计FA71/FR0/ED0/CR2487（其中两折是参考，不是训练候选）。其余四折虽然有可行模型，也没有在留出视频检出真实错误。不能把这个合计当一个完整新判断器的有效泛化成绩。

离线各视频“至少95%正确保留”的原预定目标记录：控制FA54，候选/参考FA68。必须披露两个无候选折的常数分数产生整段同分：seq05接受516正确而目标491，seq13接受297而目标283；候选合计CR2406、FR81，控制CR2367、FR120，接受的正确数量并不相等。因此不能仅用68>54独立断言精确同CR排序退化；这是含不可用参考、整同分规则的门槛记录。完整同CR/同接受数与同分界见report/CSV。没有基于结果拆散同分、修改门限或重选权重。

六项既定失败理由包含无solver候选、seq05目标/同CR比较、seq06同CR历史simple比较及总体目标无增量。按合同失败即停止，未加载VAL/TEST张量，未新增深度评价。原候选/时序/partialAUC历史增量保持，不重启辅助模型/图像头/几何/深度路线。

### 新发现与后续边界

事实：真实计数约束不必导致全接受，完整TRAIN有FR0/ED7的可行解；五个有solver解的上下文实际符号/FR预算均通过。七次均超时且最优性gap未闭合，两次无solver候选；现有控制在所有七个上下文都可表示为本轮范围内可行解。就允许的主目标而言，已有控制可行解都比本轮solver解/参考更好（详见control_feasible_witnesses.json）。

推断：当前搜索计算未有效完成，是解释本轮不可用结果必须考虑的限制；没有隔离证明模型本身达上限、错误支持不足或计数目标不适配。未知最优解可能更好也可能没有值得部署的收益，本轮不能回答。拟合能检出seq05部分错误，留出seq05却没有模型，这不是一个有效的泛化反事实。

建议：保持正式M/simple，封存本轮作为“实际计数方案固定预算未决”。不要凭本结果转去新网络/CCDB或单纯放宽FR，也不自动延长预算。若用户继续这条线，应先解决求解至少能利用已知控制可行解的问题，例如设计明确的控制可行上界约束/可复核初始化，再固定同预算求解对照；这属于下一项数值机制设计，尚未实现。本次不能直接宣称训练流程整体无问题或NP思想已经被否定。

### 工程、修复、来源与操作

本地12项测试：11通过，solver集成1项因无SciPy跳过；服务器12项全部通过，覆盖可分小例子的全局最优、相同特征相反标签无法全区分的反例、零门限/缺失、big-M包络所有系数角点、符号/整数一致性、精确FR预算、标准化、GT在线隔离及独立审核篡改检出。两端审核2558帧、36194项通过，review不重训。server两份正式前端pth SHA现场核验；本地缺两份权重明示allow-absent，未冒充检查。

首次20261014_actual_fr_v1运行中发现SciPy会原地消耗options的node_limit字段，使日志缺少固定预算；停止前full_TRAIN与seq01两次60秒求解已完成、seq05仍在求解。只核对并终止本轮PID3052868的完整命令/cwd，没有影响其他任务。首次日志/协议和failure.json保留。修复为向SciPy传options.copy()，加入库修改options的回归测试；只修日志，不改目标、FR上限、求解预算或参数。之后新目录20261014_actual_fr_fix1完成原七次计划。首次额外耗时因工程修复披露，不作为两组性能或预算对照，也没有依它调参。

源代码main 7e6cb72及修复44d8710已推送。服务器普通fetch仍被127.0.0.1:17890代理拒绝，改用双次SHA核验增量Git bundle定向导入；服务器保留codex/oof-gradient-chain-v1，由ac88be4承载最终源码，不是与main同HEAD。原有暂存patch SHA d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e前后保持；既有用户修改和上轮未提交结果整理均保留。

报告SHA aa1fe6341cde0dc928bc009b4d02ba975613d133a3496ab34ee770882a3a6c08；本轮结果集中work_dirs/port_reliability_actual_fr_v1/20261014_actual_fr_fix1/result，包含报告、逐帧、双端审核、解释、metrics.csv、analysis_summary.json和控制可行证据。未采用结果只入本运行记录，不写入论文主要成绩文档或实际论文稿。CPU、无GPU/检测推理。用户原正式policy/权重/score/输出/中心/方向保持。GT仅监督和离线计数；不导出在线新policy。

回传tar SHA58601fb0b44edfeaaa6e0fd7d8653bf2ec65053ea72e2a910dd5a3e0237f8eb3，解包限定本轮新目录、拒绝链接/覆盖；双端审核和记录一致后，清理仅本轮transport tar与两份source bundle，保留失败日志和完整结果，凭据存于结果目录。其余实验数据/缓存/权重不清理。

已完成的服务器指令，无需补跑：
```bash
cd /media/omnisky/personal_files/ljj/symEOOD
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
bash tools/run_port_reliability_actual_fr_v1.sh 20261014_actual_fr_fix1
```


## 66. 真实计数候选冻结完整VAL补评（2026-10-14）


结论：本轮冻结候选在VAL没有形成可采用收益；零门限全接受，关键高正确保留排序也弱于simple。既有TRAIN失败与求解未决保持，不能否定计数约束或NP思想整体。

完整VAL887帧、886输出、663尺寸正确、223错误、1缺失。仅使用本轮完整TRAIN实际可行模型，未使用留视频折及其回退参考。权重、训练标准化和零门限先冻结再读取VAL；未重新拟合、求解、选门限、检测推理或读TEST。

|范围|方法|FA|FR|ED|CR|缺失|接受后错误比例|
|---|---|---:|---:|---:|---:|---:|---:|
|all|formal_simple|190|10|33|653|1|22.54%|
|all|protected_simple|223|0|0|663|1|25.17%|
|all|actual_count_zero|223|0|0|663|1|25.17%|
|all|score_train95|162|57|61|606|1|21.09%|
|domain:real|formal_simple|144|10|31|189|1|43.24%|
|domain:real|protected_simple|175|0|0|199|1|46.79%|
|domain:real|actual_count_zero|175|0|0|199|1|46.79%|
|domain:real|score_train95|125|27|50|172|1|42.09%|
|domain:sim|formal_simple|46|0|2|464|0|9.02%|
|domain:sim|protected_simple|48|0|0|464|0|9.38%|
|domain:sim|actual_count_zero|48|0|0|464|0|9.38%|
|domain:sim|score_train95|37|30|11|434|0|7.86%|
|sequence:real_seq07|formal_simple|132|2|30|61|1|68.39%|
|sequence:real_seq07|protected_simple|162|0|0|63|1|72.00%|
|sequence:real_seq07|actual_count_zero|162|0|0|63|1|72.00%|
|sequence:real_seq07|score_train95|112|23|50|40|1|73.68%|
|sequence:real_seq14|formal_simple|12|8|1|128|0|8.57%|
|sequence:real_seq14|protected_simple|13|0|0|136|0|8.72%|
|sequence:real_seq14|actual_count_zero|13|0|0|136|0|8.72%|
|sequence:real_seq14|score_train95|13|4|0|132|0|8.97%|
|sequence:sim_seq10|formal_simple|46|0|2|464|0|9.02%|
|sequence:sim_seq10|protected_simple|48|0|0|464|0|9.38%|
|sequence:sim_seq10|actual_count_zero|48|0|0|464|0|9.38%|
|sequence:sim_seq10|score_train95|37|30|11|434|0|7.86%|

固定候选接受886个输出，FR归零但ED也归零；相对原policy恢复10个正确框、同时恢复33个错误框。原simple接受后错误比例22.54%，候选25.17%，接受覆盖843/887=95.04%→886/887=99.89%，全帧尺寸正确覆盖653/887=73.62%→663/887=74.75%。

同接受886时所有评分都全接受，没有排序比较意义。按原simple接受843个输出匹配，候选FA205/FR25/ED18/CR638，simple190/10/33/653；候选FA和FR均多15。按原simple CR653匹配，候选FA213（simple190）。保留全部CR663，离线候选排名FA217、simple208、score209；候选当前零门限FA223。离线阈值利用评价GT，仅用于比较，不导出在线门限。

|范围|95%正确保留目标CR|simple FA|候选 FA|score FA|
|---|---:|---:|---:|---:|
|all|630|178|197|179|
|domain:real|190|152|168|144|
|domain:sim|441|33|33|37|
|sequence:real_seq07|60|130|141|144|
|sequence:real_seq14|130|13|13|13|
|sequence:sim_seq10|441|33|33|37|

总体CR630时候选FA197、simple178、score179；Real CR190时168/152/144。总体95%是离线匹配点，不等于逐视频均受保护；另存各域各视频独立目标和完整曲线。

|范围|候选AUC|simple AUC|score AUC|
|---|---:|---:|---:|
|all|0.7092|0.7856|0.6844|
|domain:real|0.3709|0.5448|0.6648|
|domain:sim|0.5720|0.6934|0.6951|
|sequence:real_seq07|0.5655|0.5392|0.4683|
|sequence:real_seq14|0.1770|0.4152|0.4842|
|sequence:sim_seq10|0.5720|0.6934|0.6951|

局部事实仍保留：Sim保留全部464个正确框的离线候选FA42、simple45，候选局部更好；但零门限没有利用，Sim95%正确保留点FA33与simple持平。seq07 AUC .5655略高于simple .5392，但CR60时候选FA141比simple130多11；不能把AUC局部提高等同于目标工作点有效。seq14候选AUC .1770，跨视频排序仍不稳定。总体AUC .7092与Real .3709揭示 pooled 数字不能代替域内判断；当前结果不识别唯一训练根因。

建议：保留正式M＋原simple/policy，不调整当前候选门限挽救、不进入TEST。若继续真实计数方案，仅考虑在新固定数值合同中利用已核验控制可行解、获得至少不弱于既有目标的拟合结果，再用完整VAL验收；不因本轮未决结果宣布三维能力上限，也不自动追加求解预算或其他网络。

验证：8项本地测试通过，887帧、9185项评分/计数/匹配/AUC/完整曲线复核通过；另用标准库独立重建模型评分和CR630/653/663匹配计数通过。中心输出帧命中、输出覆盖、全帧正确覆盖见report，中心/方向标志及框、score、缺失身份保持。复用已审核缓存，本轮没有现场权重推理核验。

本轮本地CPU完成；没有连接服务器、Git提交或推送，也没有生成或删除压缩包。补评授权覆盖TRAIN失败后的冻结诊断，不撤回原失败，不写入论文正式成绩。


## 67. 已知FA59可行上界约束的单次完整TRAIN检验：预算内无新候选（2026-10-14）

结论：完成实现、测试、源码Git推送/定向同步、服务器CPU运行、回传与双端核验；没有产生新模型，按预定合同停止，不补出新VAL/TEST指标，不采用。该结果是固定预算搜索未成功，不是约束不可行、NP无效或三维模型能力上限的证明。保留正式M＋原simple/policy及第66节既有VAL失败事实。

### 固定设计与范围

沿用score logit、相对几何尺寸log、长宽比log三个特征、完整TRAIN拟合标准化、四系数[-10,10]、符号间隔1e-4、实际各视频FR≤floor(Ncorrect/20)，风险s<0接受。唯一算法增量为FA_TRAIN≤59（71个错误中ED≥12）的线性整数约束，目标仍最小化FA。已核验控制可行解FA59/FR25/ED12/CR2462，支持该上界；没有将它传为求解器初始解，没有声称热启动。

一次完整TRAIN求解，固定60秒、最多100000节点、gap0/presolve=True；没有六次留视频、重拟合控制、调参、重跑或新增预算。任一真实有效incumbent均冻结后评价完整VAL；若没有incumbent，返回model=null并停止，不将已知控制或全接受参考冒充新训练候选。

### 本轮实际结果

|项目|结果|
|---|---|
|SciPy/求解器|SciPy1.10.1，milp/HiGHS|
|实际耗时|60.004秒|
|终止状态|Time limit reached，status1|
|solver incumbent|无|
|报告分支节点数|0|
|新模型/新训练FA|不存在，不能填59或71|
|已知可行目标上界|59，旧控制证据|
|已证明的新最优解|无|
|完整VAL/TEST|本轮未评价，未生成新指标|

solver未给dual bound，保守目标下界为0；结合既有控制，仅能说允许域内最优FA位于[0,59]，范围太宽，不能据此判断实际可改进空间。节点数0不能独立证明预求解/数值/数据哪一环是唯一原因。

加入目标上界确实保证任何返回的有效候选FA不超过59，但不保证在60秒找到它；这次没有比原未约束求解更好的工程成果。原求解找到FA64后在VAL全接受的事实保留，不复用其VAL数值充当本次新模型结果。控制FA59是既有TRAIN保护点，不是正式原policy成绩，也不是同CR性能提升。

### 审核与来源

服务器本轮7项约束测试＋8项冻结VAL评估测试全部通过；覆盖可分/同特征相反标签的小规模最优例、真实上界矩阵方向、错误不可行报告、无incumbent不回退冒充模型、库修改options不破坏预算记录、实际FR/符号间隔、证书篡改及GT不入在线评分。相关VAL8项本地已通过且源码未变，复用先前结果；本轮新增本地7项中4通过、3项SciPy集成跳过，服务器补齐。

双端report/source/停止状态审核通过；因为没有候选，没有新的模型逐帧评价，主review independent_checks=0如实保留。另用独立描述子、TRAIN标准化重建及实际符号对2558帧控制可行解复核，两端确认FA59/FR25/ED12/CR2462、六视频FR预算及符号间隔成立，存independent_control_review.json。不能把这项控制复核写成候选性能审核。服务器两份正式前端权重SHA现场核验，本地缺两份权重明确allow-absent；没有实际检测推理。

源码main提交3b20192已推送。服务器普通fetch仍受127.0.0.1:17890拒绝影响，SHA核验增量Git bundle定向导入9个本轮所需文件（含上一轮已完成VAL补评工具），服务器保留codex/oof-gradient-chain-v1，HEAD f59a8be，非两端同HEAD/普通pull。原暂存patch SHA d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e保持；本地既有未提交文档和其他源码保留。

结果路径：work_dirs/port_reliability_actual_fr_bound_v1/20261014_actual_fr_bound_v1/result。report SHA f8bed8c8ca93522bbfa9d5ad06b3163c002e48d684e63ca607efe672bf41c66e。回传tar SHA07ce2c4896dde180f5916bef83e64e5ea5000f0faa85ffcbdcca959d5c1f2aab；安全解包只限本轮新目录，拒绝链接与覆盖。结果和日志保留，记录同步后仅清理本轮transport tar与source_3b20192.bundle，清理凭据另存。没有清理其他版本；未写论文主要成绩。

### 建议

结束当前SciPy固定预算的上界约束试验，不自动延长预算或继续换约束。已有simple控制可行，但本轮求解没有有效利用它；若未来继续整数优化，需另立真正支持并核验初始可行解的数值实现合同，先确保搜索能从已知控制开始。即使解决求解，跨视频VAL排序与决策迁移仍须独立验收。当前不据此转网络、CCDB或重启辅助模型，正式M/simple保持。

已完成的运行指令（无需补跑）：
```bash
cd /media/omnisky/personal_files/ljj/symEOOD
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
bash tools/run_port_reliability_actual_fr_bound_v1.sh 20261014_actual_fr_bound_v1
```


## 68. 完整TRAIN类别条件跨视频偏差与共同支持检验（2026-10-14）

结论：发现固定simple相对尺寸、比例贡献在seq05/seq06中的描述性错误关联方向不同；但类别条件联合分布共同支持很少，本次不足以支持启动CCDB式权重训练。不是新判断器，没有FA下降的性能结果，不证明唯一根因或CCDB理论无效。正式M及simple/policy保持。

### 固定范围与计算

复用完整TRAIN2558输出、2487正确/71错误；标签来自交付框与GT的真实双边≤10%，不用旧接受标志代替。仅score logit、相对几何尺寸log、长宽比log。全TRAIN原始特征按linear .25/.5/.75确定固定四分位边界，等于边界进右桶；联合4×4×4=64格。贡献c_j=w_j(x_j−mu_j)/scale_j固定使用formal simple系数/标准化，无翻转、拟合、门限更新或新权重。

按类、按视频分别报告原始分布/直方图、贡献AUC、错误-正确贡献均值差；逐类pair marginal与joint total variation，仅描述有限样本分布差异。共同支持要求同一个三特征联合格内两个视频都同时有正确和错误样本。>=10错误仅为预定保守描述比较条件，不是显著性检验或独立样本数保证；所有视频结果保留，少错误视频不作稳定方向结论。按连续帧/采集边界数错误段，不等于独立采集事件。

### 实际结果

|视频|正确/错误|错误连续段|score贡献AUC|相对尺寸贡献AUC|比例贡献AUC|
|---|---:|---:|---:|---:|---:|
|real_seq01|338/1|1|0.7840|0.0059|0.0178|
|real_seq05|516/44|28|0.5906|0.4009|0.6468|
|real_seq06|453/13|10|0.5704|0.6363|0.4682|
|real_seq12|136/5|5|0.4824|0.6382|0.4559|
|real_seq13|297/7|6|0.3785|0.4993|0.4916|
|sim_seq08|747/1|1|0.9116|0.0750|0.8581|

唯一两边均>=10错误的Real配对为seq05/seq06：相对尺寸贡献AUC .4009/.6363，比例 .6468/.4682，方向在.5两侧；score .5906/.5704同向。这里只是描述性方向差异，没有显著性结论、特征因果或偏差因素认证，seq06比例仅略低.5。

该配对正确类joint TV .8580、错误类 .8864；正确类共享11格、覆盖226/516与76/453帧；错误类共享2格、覆盖5/44与3/13错误。两类两视频同时存在的共同2格：seq05正确65/错误5、seq06正确8/错误3；两边共8个错误，仅占两视频57个错误的14.04%。共同正确样本只有73帧。不要从单轴边际重叠推出联合可比。

全部Real配对中，seq06/13有两格四方支持（正确127/24，错误4/2），seq12/13一格（正确19/10，错误1/1），其余七个Real配对无四方联合支持。后两组包含不足10错误的视频，不能作为稳定权重学习依据。0个共同格不是“特征完全无用”，分箱粗糙、样本稀疏、误差类型不同均可能参与。

### 解释与下一步

此前“分数/工作点对视频敏感”的诊断获得更具体的分布和贡献线索；但这次不能证明相对尺寸/比例是伪相关特征，更不能直接移除（历史消融已经失败）。几何特征可能同时携带真实质量信息。类别平衡不能消除类别条件失配，但这个结论也不意味着任何CCDB权重一定有效。

目前不开展CCDB式训练：只有一个满足支持规则的Real配对，且其共同错误支持仅5/3帧，不能据此让少数样本承担跨视频平衡。没有实际学习权重，因此不能报告权重ESS、声称已证明权重会失控，或宣布加权在数学上不可行。本次仅表明拟议前提证据不足，未额外设检验门槛或调分箱来制造通过。

建议收束当前三维重加权前提检验，不自动训练或增加诊断分箱/样本权重搜索。后续若继续可靠性，需要提出能辨别稳定错误的新观测证据或可验证且不同于旧路线的机制；本次不指定已有效的新结构，也不重启辅助检测器/ROI/时序或修改检测框。现有正式结果可用于描述有限适用范围；不要因为此诊断再次声称无法优化。

### 工程、记录和清理

本机和服务器各8项测试通过，涵盖真实标签与接受状态分离、固定贡献方向、同分右桶、联合四方支持、连续段采集边界、稀疏视频、无修改及报告篡改检出。2558帧两端独立标量描述子/标签/联合桶逐帧核验及全部统计/pair重放通过；重放不是独立统计保证，无显著性/bootstrap或新训练。服务器两份正式权重SHA现场核验，本地缺两份权重明确allow-absent；前端/缓存/框/score/三标志及缺失身份保持，GT仅离线，本TRAIN无缺框。

源码main78413d1已推送。服务器fetch仍受127.0.0.1:17890失效代理拒绝影响，经SHA核验bundle仅导入4个新文件，保持codex/oof-gradient-chain-v1，server HEAD7fa7d57；不是两端同HEAD或普通pull。既有暂存patch SHA d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e保持，本地旧未提交文档/其他代码保留。

结果统一work_dirs/port_reliability_class_condition_v1/20261014_class_condition_v1/result。report SHA897c250920e8ae8347686463801829ce037d3fce5e257582d0092d050fa9ee2f；tar SHA903a4c08c4f68025837e0871f83ffd33e960700ddc395ba442ca02054aa1e464。只解包本轮新目录，拒绝链接/覆盖，双端核验与记录后仅删除本轮transport tar/source bundle。完整统计/日志保留，不复制新逐帧缓存、不改论文主要成绩。

已完成指令，无需补跑：
```bash
cd /media/omnisky/personal_files/ljj/symEOOD
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
bash tools/run_port_reliability_class_condition_v1.sh 20261014_class_condition_v1
```

## 69. 实验C：冻结尺寸错误严重程度与长短边类型分析（2026-10-15）


状态：描述性分析完成，无新判断器、policy或性能提升声明。

### 固定合同与来源

固定M=B24＋σ1.5/epoch03及cachefix1 simple；只读取完整TRAIN2558和VAL887的保存结果。TEST未读取，无拟合、检测推理、显卡或门限更新。最大绝对双边相对误差e：≤10%正确，(10%,15%]轻微、(15%,25%]中等、>25%严重。长短边规范化，原始三标志逐帧API重放一致。

原policy复现TRAIN FA13/FR1037/ED58/CR1450；VAL FA190/FR10/ED33/CR653/MISSING1。

### 原policy严重程度分布

|范围|错误层|错误数|FA|ED|检出率|FA误差中位/P95/最大|
|---|---|---:|---:|---:|---:|---|
|TRAIN|10–15%|63|13|50|79.37%|11.10%/12.78%/13.05%|
|TRAIN|15–25%|7|0|7|100.00%|—/—/—|
|TRAIN|>25%|1|0|1|100.00%|—/—/—|
|VAL|10–15%|146|133|13|8.90%|11.49%/14.13%/14.94%|
|VAL|15–25%|66|53|13|19.70%|16.34%/20.65%/23.15%|
|VAL|>25%|11|4|7|63.64%|48.97%/50.30%/50.46%|

### VAL视频内AUROC

每层错误与同视频全部正确尺寸框比较；排除其他错误层。表中为simple/score。无严重错误时记—；少于10错误只描述。

|视频|轻微|中等|严重|
|---|---|---|---|
|real_seq07|0.4840/0.4446 (n=95)|0.5611/0.4547 (n=57)|0.9381/0.7698 (n=10)|
|real_seq14|0.3358/0.3644 (n=9)|0.5938/0.7537 (n=4)|—|
|sim_seq10|0.6653/0.6788 (n=42)|0.8681/0.7716 (n=5)|1.0000/1.0000 (n=1)|

### VAL错误类型

|范围|仅长边 错误/FA|仅短边 错误/FA|双边 错误/FA|
|---|---|---|---|
|all|73/59|67/63|83/68|
|sequence:real_seq07|32/20|51/47|79/65|
|sequence:real_seq14|10/10|1/1|2/1|
|sequence:sim_seq10|31/29|15/15|2/2|

### 发现与解释边界

1. VAL190个FA中133个(70%)是10–15%，53个(27.89%)是15–25%，4个(2.11%)超过25%。不是全部轻微边界错误；严重FA全部在seq07，最大误差约50.46%。
2. VAL仅长边FA59、仅短边63、双边68；涉及短边的FA为131/190。按超限轴方向，174/190属于仅长偏小、仅短偏小或双边共同偏小。GT方向只用于离线描述，不能进入在线门控。
3. seq07：轻微FA83/95、中等45/57、严重4/10。128个10–25%的FA是偏小类型，另4个严重FA是双边偏大。轻微/中等simple AUROC .4840/.5611，严重.9381。不能用整体严重错误易识别替代主要错误群的辨识。
4. seq14的12个FA均为偏大类型，Sim的46个FA均属于偏小类型；不能将统一偏小规则用于全数据，亦不能据GT或视频名在线路由。
5. TRAIN只有8个>15%错误(7中等、1严重)，都来自seq05；Sim TRAIN仅1个轻微错误。它描述错误支持集中，但不证明唯一根因。TRAIN原policy拒绝了全部8个>15%错误，同时FR1037；同总体95%正确保留时仍接受3个>15%错误(含唯一严重错误)，不能拿原点高检出率称泛化成功。
6. VAL pooled分层AUROC .7390/.8565/.9793，而Real .4918/.5729/.9241；域/视频组成明显影响总体值。严重只有11个，不作统计显著性或理论不可分结论。
7. 离线same-CR并非重放旧门限：seq07正式CR61/FA132，simple按最小整tie阈值匹配CR61为FA131；95%正确保留CR60为FA130/FR3。Sim保护全部CR464时simple FA45对原点46；这是同一排序去掉不必要尾部的离线参照，不是新policy或泛化收益。分组参考门限不能相加当一个可部署全局策略。

### 下一步建议

优先将待改问题限定为：怎样给10–25%偏小错误提供跨视频可用的判别依据，同时保护正确框及seq14偏大类型。当前三维在seq07的这部分排序接近随机，单调校准、重新切严重程度区间或统一收紧门限缺少支持。先对照TRAIN已有这些具体误差类型与原特征支持，再说明一个此前没有的信息来源；不自动重开PQA/ReDC/OOF/时序、增加权重或进行新训练。不放宽10%标签。此分析未隔离数据、特征或学习目标的唯一根因。

### 工程审核与结果身份

本地/服务器各13项测试通过；Python3.8与bash语法、来源闭包及输入SHA检查通过。服务器两份正式前端权重现场SHA核验，本地缺权重仅按明确allow-absent审核保存证据。两端逐帧标量误差、区间/类型/四状态、成对AUROC、同CR/同接受数/同分界和完整报告重放通过：3445帧、3745项独立计数检查（另含完整统计重放）。不称统计独立或GPU复现。
源码main1896581已推送；服务器因127.0.0.1:17890代理失败，通过核验Git bundle仅导入4个新增文件，server HEAD633c312、原分支codex/oof-gradient-chain-v1保留，历史暂存patch SHA d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e保持。不是两端同HEAD。
report SHA bee8c6f825428303b99e415866018cad28b002bd51b0371f5ec4ede9ed4a2a78；分析包SHA fd8804a1b0f42f66fe75a5b4c75ff8f9f4e66488fcbfdbdedbfed4d707cacea8。完整结果/日志保留；双端核验和记录后仅清理本次tar与source_1896581.bundle，不修改其他结果。

已执行指令（无需重跑）：
```bash
cd /media/omnisky/personal_files/ljj/symEOOD
source /media/omnisky/4435947f-033c-4e45-904d-f2c4d0bfbf70/Anaconda3/etc/profile.d/conda.sh
conda activate mmrotljj
bash tools/run_port_reliability_severity_v1.sh 20261015_severity_v1
```


## 70. PCR适用前提核对：正式M为无NMS的Top-1（2026-10-15）

用户授权候选支持代码及服务器流程后，本地静态核对发现原设计前提不成立；随后在服务器项目目录、mmrotljj环境解析正式B配置并通过inspect读取实际导入的SymEOODHead方法，确认同一行为。此轮仅配置/加载源码核对，没有构建权重推理、采集TRAIN候选或训练判断器，没有VAL/TEST访问，也没有新性能指标。

事实：正式配置为crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py，SymEOOD/SymEOODHead，score_thr=.05、max_per_img=1。虽然test_cfg保留nms.iou_thr=.1及nms_pre=2000字段，实际simple_test传with_nms=False，_get_bboxes_single经有效区域及score筛选、框解码后直接scores.topk(max_per_img)，未调用NMS；该方法也没有使用nms_pre进行候选截断。配置含有NMS字段不能证明运行NMS。

纠正前一轮方案：正式主框不存在NMS抑制候选集合，不能按原NMS抑制关系分组，也不能把Top-1未选框称为NMS丢弃框。PCR原论文的pre/post-NMS统计与数据集级性能估计不能直接迁移为最终M双边≤10%的判断器。

推断与待验证：可另行研究Top-1之前的候选分歧，但一对一训练的非优胜位置不是独立质量测量，其分数/回归质量及局部候选支持数量都需核实；当前未采集，不能声称候选足够或不足。若未来开展，须明确这是新适配方案，保留原score筛选与最终Top-1，不增加NMS、不放低score搜候选、不用候选替换主框；首先核实可用支持，再决定是否值得完整TRAIN分析。

本轮结论：原PCR/NMS候选方案前提不适用，在接口核对处停止；没有算法失败或可靠性改善结论。正式M、simple、三个标志及旧实验保持。没有为无效前提追加GPU运行、提交模型改动或生成传输压缩包。


## 71. 旧原图边界读取：固定36帧TRAIN可视核对（2026-10-15）

状态：VISUAL_REVIEW_COMPLETE_TEED_PREMISE_NOT_ESTABLISHED。完成本地代码/自审、Git推送、服务器mmrotljj CPU运行、原图/剖面/标志双端独立核验及可视分析。没有TEED加载、检测器推理或判断器训练，没有VAL/TEST访问，无新性能收益。

范围：完整TRAIN2558帧中先固定正确、10%～25%偏小、偏大/混合三组各12帧，按视频尽量等额再按帧序均匀抽取。三组可选2487/35/36，偏小组仅4个Real视频，无Sim；不从VAL补样本。选中旧simple状态分别CR8/FR4、FA4/ED8、FA6/ED6。这些不是新模型成绩或总体估计。

实现：保留旧7剖面/侧、129点、±1原像素梯度和同峰中位规则，生成36张1800×1120原图/四侧曲线图及6张联系图。GT仅用于离线抽样及矩形交点，不等于真实可见边界；原图放大不增加有效分辨率。图隐藏分组/误差数字/标志，但审读者知道设计，不称严格盲审或新增人工标注。在线profiles仅接收灰度和预测框。

核验：本地/服务器各8项测试通过；36份原evidence精确还原；1008条剖面用独立标量双线性插值/选峰复核，梯度最大差0；108分量标志保持；独立重建抽样。36原图SHA、46初始制品SHA、30输入/源码pin通过。独立脚本与双端mac_review/server_review保存在结果目录。完整TRAIN旧FA13/FR1037/ED58/CR1450保持。

发现：多个视频存在内部结构、连接杆、外围框架多峰竞争，正确框也有；17帧/34轴在本次审读中边界难辨，均保留B+U，其余保留U。仅看到干扰峰不足以确认存在清晰且唯一的目标替代边界，所以A只列候选、未确证；C/D也未确证。real_seq05_00389/_00455规范长轴差约47.88°/36.88°，固定M轴读出混入方向差异。1008采样线中8条GT出口在搜索范围外或无交点，涉及4分轴；未扩大范围。不能把这些描述推断为标注错误、像素信息绝对不足或TEED必然无效。

决策：本轮尚不足以支持“仅换TEED边缘响应即可解决尺寸判断”的前提，停止自动推进；不拟合新头、不增加范围或样本搜索。未来若检验冻结TEED，需先明确可辨认且对应标签范围的几何证据，并保留正确框对照。边缘更清晰不代替FA/FR验收。

源码main f77ab2d已推送；服务器Git代理127.0.0.1:17890失败，核验bundle仅导入4新增文件，原分支codex/oof-gradient-chain-v1提交50a0316，既有暂存patch SHA d54105b5f8055df157511d1d0af99e065bd2b61153373a69d377167d653eae2e保持。不是两端同HEAD。完整结果位于work_dirs/port_reliability_boundary_readout_review_v1/20261015_fixed36_v1/result，analysis.md与visual_review.json给出逐帧依据，未写论文主要结果。

报告SHA 9fddd1d4456cfa7befa275a2a352bb752e672c443a3af2245af617d1349a6945；传输包SHA 6dc57684c99bea8a35d330bf542310a586983af1a4c36a204a9cdd7bd204b2d1。记录同步后仅清理本轮准确SHA匹配的传输tar及source_f77ab2d.bundle；完成以双端cleanup_receipt为准，完整结果保留。

已运行，无需补跑：
```bash
bash tools/run_port_reliability_boundary_readout_review_v1.sh 20261015_fixed36_v1
```


## 72. 已有判断器配对排序核对：容量、迁移与冻结工作点（2026-10-15）

状态：SAVED_RESULT_AUDIT_COMPLETE_NO_AUTOMATIC_ADVANCEMENT。用户授权代码实现、指定服务器运行、回传复核及沿用既有收尾流程。本轮只复用已有完整TRAIN/VAL评分，不训练、不提取特征、不做检测推理、不读取TEST、不生成新policy、不搜索校准。正式M=B24＋midpoint sigma1.5/epoch03及simple保持。已暴露TEST的历史结论仍留原记录，不用于本轮选型。

### 可比性与来源

五组各3445帧（完整TRAIN2558、VAL887），共17225条来源记录；四组对首组13780次逐帧核对，原框、GT、样本身份及原三标志精确一致。标签均按最终M规范长短边相对误差各≤10%计算；TRAIN71错/2487对，VAL223错/663对/1缺失。所有五组均有逐帧评分；未找到相同输入、材料、监督、训练合同而仅增加容量的严格配对，不补跑缺失配对。

|配对|原结果目录（work_dirs下）|可回答的问题与限制|
|---|---|---|
|linear / ReDC|port_reliability_redc_size_v1/20261008_redc_v1/result|相同可用291维描述量、材料、平衡BCE、同批次1000步；292与4802参数，同时改变线性/正温度受限读出，不能隔离纯容量|
|temperature / signed residual|port_reliability_readout_compare_v1/20261008_readout_v1/result|相同材料、特征、4802参数、预算，读出公式不同；读出对照|
|midpoint / native MLP|port_reliability_feature_source_v1/20261009_source_v1/result|相同材料、4290参数、目标与预算；原始特征来源及各自TRAIN PCA改变，属于来源对照|
|BCE / within-video rank|port_reliability_within_video_rank_v1/20261009_within_video_rank_v1/result|相同native输入、容量、初始化、更新数；增加排序目标，反向成本不同，属于目标对照|
|coarse BCE / spatial BCE / spatial axis|port_reliability_spatial_axis_v1/20261009_spatial_axis_v1_fix2/result|A/B改变空间表示；B/C增加连续分轴辅助监督；总参数均13115，BCE实际更新13081、axis13115，不是纯容量配对|

各组均为末态epoch100、相同seed1701批次顺序；源合同、参数数量、归一化及优化器/加权差异保存在manifest和report.original_contracts。simple输入仅3维，不能将上述整个候选相对simple的收益归因为容量。旧B geometry/ROI/structure MLP前端不同，排除成绩拼接；其旧合同仅作本地归档元数据，不要求服务器存在、不读取旧TEST。native旧审查器当前SHA与原运行SHA不同，已披露，未重写旧来源清单或生成新评分。

### 冻结点与离线排序

原冻结门限同时应用TRAIN/VAL。完整总体/分域/逐视频FA、FR、ED、CR、MISSING、接受/输出覆盖、正确保留、全帧正确覆盖、最长正确FR、AUROC见metrics.csv。双方冻结点均作为锚点比较同接受数与同CR，见matched_points.csv；同接受数按身份拆同分并报告FA上下界，同CR接受整个cutoff tie且标记实际CR超目标。匹配门限只用于离线诊断，不写回policy。

完整整tie风险—覆盖/正确保留曲线134492点保存于curves.jsonl.gz。95%—100%正确保留只是本轮描述区间，穷举该区间每个整数CR，不是新增保护合同、独立重复试验或门限选择。单类不报AUROC，错误<10的组仅描述；TRAIN多组只有1、5或7个错误，帧/配对/工作点均相关，不据大配对数宣称统计显著。

|方法|TRAIN FA/FR/ED/CR（冻结点）|VAL FA/FR/ED/CR（冻结点）|TRAIN / VAL错误AUROC|TRAIN CR2363 / VAL CR630 的FA（离线）|
|---|---|---|---|---|
|simple|13/1037/58/1450|190/10/33/653|.801752 / .785626|48 / 178|
|同输入linear|23/192/48/2295|206/12/17/651|.918698 / .608607|35 / 195|
|ReDC / 重现temperature|71/0/0/2487|201/3/22/660|.850864 / .667519|68 / 179|
|signed residual|47/2/24/2485|219/23/4/640|.996506 / .599267|1 / 218|
|midpoint来源MLP|1/0/70/2487|206/3/17/660|1 / .734648|0 / 170|
|native来源MLP|0/0/71/2487|183/10/40/653|1 / .716610|0 / 161|
|native BCE控制|0/995/71/1492|190/11/33/652|.956823 / .786032|14 / 170|
|within-video rank|1/938/70/1549|191/4/32/659|.951353 / .797239|14 / 159|
|spatial BCE|1/991/70/1496|202/6/21/657|.943469 / .783522|14 / 180|
|spatial axis|1/994/70/1493|201/6/22/657|.946063 / .784192|14 / 179|

表中两角色离线CR分别为ceil(.95×2487)=2363与ceil(.95×663)=630；这两个点不替代完整曲线和逐视频分析。coarse BCE、score及全部分组结果保存在完整报告，未因不列入便读表而删除。

### 收益在哪一环节消失

1. ReDC高保留区间TRAIN不及同输入linear，VAL却优于linear；相对simple仅局部，seq07高保留全区间较差。不能归为纯容量增益，也不能把某个低CR点的TRAIN收益推广到高保留区间。
2. signed residual相对temperature在TRAIN高保留区间125个整数CR目标全优，VAL34个目标全劣，是明确的本次读出增量迁移反转。不能因TRAIN AUROC接近1而继续扩大网络。
3. native与midpoint TRAIN排序均完美；native的VAL总体局部优势仍在：同CR653时FA183 vs simple190，同836接受FA183 vs188；不是VAL全无增益。但seq14同CR128为13 vs12，Sim同CR464为48 vs45；Sim在95%—100%区间24个目标均较差。不能只以pooled或AUROC判采用。
4. within-video rank相对BCE的TRAIN高保留区间无新增优势（53平/72劣）；VAL总体多数点改善，相对simple同CR653为FA183 vs190。但seq07同CR61为139 vs131、Sim同CR464为48 vs45，优势未稳定覆盖实际保护要求。目标对照不能改写成容量对照。
5. spatial axis相对spatial BCE在TRAIN有局部增量，VAL有部分保留；相对simple的VAL高保留区间仅2优/6平/26劣，没有形成正式升级。

本轮结论：没有可隔离“仅增加容量有效”的严格证据；也不能说所有复杂候选在VAL都丢失排序优势。当前有局部可利用信息，但尚未证实兼顾逐视频保护的稳定候选，不把问题统一归为门限失配。原逐域/视频95%保护及晋级合同已检验失败，不换“校准”名称重复搜索；不自动采用、扩大容量、校准搜索或追加TEST。未来若实际使用要求改变，应先明确保护/FA目标与代价，再讨论单个冻结评分的有限设计，而非事后改门槛放行。

### 工程、运行、核验与收尾

本地代码提交5d584b4，Python3.8兼容修复ebf9cbd。首次服务器20261015_saved_pairs_server_v1在最终source-pins序列化处因相对__file__报错，没有写出正式报告；保留failure.json。修复只将脚本路径resolve并补回归测试，材料/指标不变，另RUN_ID重跑。本地与服务器最终各11项测试通过；Python3.8语法及bash语法通过。两端195组指标、全部134492曲线点精确一致；内置独立标量重算7419789对错误/正确评分及整tie曲线计数，重现原冻结四状态。数字表示计算核验量，不表示独立样本数或性能提升。

服务器实际目录/media/omnisky/personal_files/ljj/symEOOD，环境mmrotljj；CPU-only，未加载权重、未训练、未推理。本轮初查服务器为codex/oof-gradient-chain-v1/50a0316；同步前其他工作已推进到main/25a5a70。本轮保留现场main，定向导入2638854、修复6cd3275，没有切换/重置分支；不再把历史分支与暂存指纹当当前事实。导入前后及运行前后既有工作区diff SHA134f38b44b5f250e1aa27986e49a2283727baa26b73689677aee609228224bf6、暂存diff SHAe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855保持。之后对本节的追加属于授权记录更新，不覆盖双方不同的旧文档。

GitHub origin/main推送被自动审批拒绝：未明确授权向默认分支发布，且可用更窄bundle同步。没有绕过或重试推送；用SHA核验bundle完成指定服务器同步，两端不是相同HEAD。原用户文档已有未提交内容保持，仅追加本节，不将整份草稿混入源码提交。

正式结果统一：work_dirs/port_reliability_paired_ranking_audit_v1/20261015_saved_pairs_server_v1_fix1/result，analysis.md为便读表，report.json为全分组/匹配/合同，另有metrics.csv、matched_points.csv、完整曲线、independent_review与mac_review。84项输入pin及5项本次源码pin通过；回传16成员包安全解包、原completion逐文件SHA通过。report SHA121e8fbdb42691754e6bb4c7ac353847b5ce70af1e0d77fd45f833f82e192eb3，transport tar SHA74fbb496103aa07d367c4e1c822778c6b5f78579c47037cc3145f6a904677533。

双端核验与本节同步后，仅清理本次tar及两个源码bundle；实际删除以result/cleanup_receipt_local.json与cleanup_receipt_server.json为准。完整结果、首次失败、同步回执、旧实验、正式policy和用户已有改动保留。不修改论文或主要结果文档。


## 73. 冻结候选VAL/TEST总表与论文对比职责（2026-10-15）

用户授权汇总已有VAL与TEST并询问论文是否需要可靠性对比。本轮仅本地读取已保存报告与SHA核验，无训练/推理/重新评分/校准/远程运行，无新policy或基线替换。主要资料集中追加结果文档第14节（docs/obb/可靠性实验主要结果与对比分析_20261006.md），同目录交付CSV、配对CSV和来源JSON；不另存原始逐帧结果或回传包。保留现有用户未提交内容，未Git提交或推送。

综合结论：不能仅凭第72节VAL推荐native替换正式版。native VAL同CR653 FA183 vs190，但冻结TEST313/33/29/1056（FA/FR/ED/CR），相对正式275/34/67/1055多38FA、仅多1CR；同1369接受simple292，同CR1056 simple275。最初时序VAL同785接受155 vs170有价值，但TEST同1250接受244 vs239反转。部分AUC/修正回退TEST308/14/34/1075，较正式多33FA、少20FR；比原四维仍少2FA/2FR。midpoint来源MLP另有TEST局部优势274/28/68/1061，同1335接受simple276，同CR1061 simple279；VAL206/3/17/660且TEST seq03正确保留76/91，不据TEST反选此控制。视频内排序仅TRAIN/VAL完成，TEST缺失明确标注，未补跑。

总表区分原正式simple门限0.41665113433410333与来源/读出实验full_simple门限0.43131551898452714；固定score亦区分原风险门限0.7043236196041107与实验0.7688139975070953。同数/同CR为离线诊断，不产生部署门限。允许局部取舍，不要求所有指标都改善，但明显TEST排序反转不能写成稳定降FA。正式M/simple身份保持。

论文职责：可靠性若作为贡献，需要固定同M的raw/score/simple/采用候选比较、同数/同CR与覆盖、三分量及分域/视频结果；所称新增机制需直接消融，真正采用的模块需效率。无需把每个历史失败候选当正文创新。小论文不报告深度时，不据可靠性标签宣称深度/摇摆角收益。中心只是有效输出保留，不是学到中心错误判断器。详细来源与会计核验见第14.9节，不冒称本轮重新逐帧审计或服务器执行。

本轮资料核验完成：180行实际工作点、144行匹配对照均回查原报告JSON，7份来源SHA保持，整数与比例/分视频加和通过；CSV来源键使用转义JSON Pointer。git diff --check通过。不是新训练或逐帧重推理的性能验证。
