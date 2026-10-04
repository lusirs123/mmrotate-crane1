# SymEOOD＋尺度增强B：当前分量可靠性交接

更新：2026-10-04；TEST主结果仍为服务器`port_simple_reliability_v1_test_review_20261003_205844.tar.gz`。新增既有TRAIN/VAL冻结评分的CPU可分性检查，第8节为实现/本地结果，第9节为服务器回传核验与判读。

本文只维护当前B＋simple v1模型、协议、服务器结果和下一步。用户要求旧流程/旧成绩从当前正文删除并替换为新结果，本次已重写；旧记录在[替换前归档](archive/20261003_reliability_replaced_by_simple_v1/README.md)保留，原始实验文件和模型未删除。检测几何优化继续维护[独立几何交接](geometry_precision_handoff_20261001.md)，本轮未改其内容。

## 1. 当前决定与模型

固定前端SymEOOD＋等比例尺度增强B，VAL选择epoch24；可靠性为`port_simple_component_reliability_v1`，不是ROI/结构分支。

当前模型不使用DINO、候选融合、Base V3/V5.1、历史保持或缺测预测。连续状态、物理摆角、深度/空间风险及报警接口留给大论文。

```text
当前RGB → 冻结B原图OBB/score → 中心/尺寸/方向三个独立使用标志 → 分量质量与覆盖评价
```

中心标志保留每个有效B输出，不另训练中心正确性预测器。尺寸、方向各为独立线性风险模型，输入只有当前框的score logit、相对几何尺寸和长宽比；共8个拟合系数。原图长短边只在风险计算中规范化，导出的原宽高、角度、中心、score及输出数量保持不变。

TRAIN2558帧（real1810、sim748）用于拟合/标准化；VAL887帧（real375、sim512）用于此前固定的95% pooled覆盖规则产生单一全局工作门限，之后冻结。real原生轴线转换OBB和sim Webots生成OBB提供监督；sim没有轴线标注，不要求补轴线。GT不进入在线API。尺寸/方向平衡类sigmoid是风险评分，不是校准错误概率。

三特征为`logit(score)`、`.5*(log L+log S-log W-log H)`、`log(L/S)`，L/S为预测规范长短边，W/H为原图尺寸。固定正负类各总权重0.5、L2=0.1、float64 Newton最多100次/梯度容差1e-8；本次TEST不再拟合或调整这些参数。

## 2. 输入输出与评价协议

TEST1440帧：real_seq03 200、real_seq04 668、sim_seq09 572。缺失9帧全部保留，原输出1431帧。本版TEST全部1440帧方向可评。

中心正确为原图欧氏误差<15px；尺寸正确为规范化长短边最大相对误差≤10%；方向正确为长边π周期角误差≤3°。GT aspect≥1.2仅用于离线方向评价，不用于在线接受决策。中心命中率仅以输出帧为分母，另报输出覆盖和全帧中心正确覆盖。

`raw`保留所有有效B分量；`score_only`用冻结VAL门限筛选尺寸/方向、中心仍全部保留；`simple`用两组冻结线性风险与VAL门限分别判尺寸/方向、中心仍全部保留。无B输出时三个标志均false。GT只在评分后定义正确/错误，不用于在线判断。

同接受数score只作离线排序诊断：匹配simple数量，不能把它称为部署门限或由TEST校准的score。固定score门限与同数score对照是两种不同职责。

标志是模型的建议使用/拒绝决策，不是已知GT标签。在线输入仅`pred_original`与`[W,H]`；GT、domain、sequence、历史或未来帧都不进入API。保留等比例变换和原图坐标链；尺寸/方向拒绝不删除中心，接口保持不等于深度精度得到保证。

## 3. 当前服务器TEST主结果

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
