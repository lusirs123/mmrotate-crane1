# SymEOOD 分量可靠性交接：固定B＋正式midpoint基线

更新：2026-10-04；用户已固定B24＋正式midpoint23为后续可靠性基线。第14节是迁移及14图模板读取的服务器回传结论，第15节是固定模板在432参考留出TRAIN＋887 VAL上的全量验证入口，GPU结果待服务器运行。现有TEST主结果仍是B＋simple v1的`port_simple_reliability_v1_test_review_20261003_205844.tar.gz`，不能迁为midpoint成绩。第8/9节为冻结评分可分性检查，第10/11节保留尺寸参考v1原结果。

本文维护已验证B＋simple v1证据，以及用户本轮要求的B＋正式midpoint可靠性迁移。此前替换的旧流程在[替换前归档](archive/20261003_reliability_replaced_by_simple_v1/README.md)保留，原始实验文件和模型未删除。检测几何优化维护[独立几何交接](geometry_precision_handoff_20261001.md)，本轮没有修改该窗口的代码/记录。按最新偏好，后续服务器结果不复制为本地复核目录，仅在本文记录必要结论。

## 1. 当前决定与模型

当前前端固定SymEOOD＋等比例尺度增强B（VAL epoch24）＋正式midpoint（原完整VAL规则选head_epoch_23），接口为`MidpointReliability`包装现有simple三标志。迁移已完成服务器collect/fit/smoke并回传复核（第14节）；其尺寸/方向判别尚未证实稳定优于score。用户本轮授权固定参考epoch04和已实现模板读取，完成一次全量TRAIN留出/VAL效果验证（第15节）。下述第2—11节B指标保留原协议身份，不等于新前端结果。

当前模型不使用DINO、候选融合、Base V3/V5.1、历史保持或缺测预测。连续状态、物理摆角、深度/空间风险及报警接口留给大论文。

```text
当前RGB → 冻结B24＋正式midpoint23的原图OBB/score → 中心/尺寸/方向三个独立使用标志 → 分量质量与覆盖评价
```

中心标志保留每个有效最终midpoint输出，不另训练中心正确性预测器。尺寸、方向各为独立线性风险模型，输入只有最终框的score logit、相对几何尺寸和长宽比；共8个拟合系数。原图长短边只在风险计算中规范化，可靠性计算保持最终宽高、角度、中心、score及输出数量不变。当前simple使用第14节新前端重新拟合后的policy，旧B policy仅作历史对照。模板仍是尺寸候选，未替换在线标志。

TRAIN2558帧（real1810、sim748）用于拟合/标准化；VAL887帧（real375、sim512）用于此前固定的95% pooled覆盖规则产生单一全局工作门限，之后冻结。real原生轴线转换OBB和sim Webots生成OBB提供监督；sim没有轴线标注，不要求补轴线。GT不进入在线API。尺寸/方向平衡类sigmoid是风险评分，不是校准错误概率。

三特征为`logit(score)`、`.5*(log L+log S-log W-log H)`、`log(L/S)`，L/S为预测规范长短边，W/H为原图尺寸。固定正负类各总权重0.5、L2=0.1、float64 Newton最多100次/梯度容差1e-8；旧B与新midpoint遵循同一拟合规范，本轮全量读取验证不重新拟合参数。

## 2. 输入输出与评价协议

本节及第3—9节TEST分母/成绩属于旧B＋simple v1协议；当前midpoint只读取TRAIN/VAL，最新已测结果见第14节，全量候选入口见第15节。

TEST1440帧：real_seq03 200、real_seq04 668、sim_seq09 572。缺失9帧全部保留，原输出1431帧。本版TEST全部1440帧方向可评。

中心正确为原图欧氏误差<15px；尺寸正确为规范化长短边最大相对误差≤10%；方向正确为长边π周期角误差≤3°。GT aspect≥1.2仅用于离线方向评价，不用于在线接受决策。中心命中率仅以输出帧为分母，另报输出覆盖和全帧中心正确覆盖。

`raw`保留所有有效B分量；`score_only`用冻结VAL门限筛选尺寸/方向、中心仍全部保留；`simple`用两组冻结线性风险与VAL门限分别判尺寸/方向、中心仍全部保留。无B输出时三个标志均false。GT只在评分后定义正确/错误，不用于在线判断。

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
