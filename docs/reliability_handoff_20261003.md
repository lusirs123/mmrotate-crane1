# SymEOOD＋尺度增强B：当前分量可靠性交接

更新：2026-10-03；最新证据为服务器`port_simple_reliability_v1_test_review_20261003_205844.tar.gz`。

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

本版policy/门限/B及新TEST报告保留冻结，当前流程收尾完成。下一项先复用既有TRAIN/VAL原框、GT与冻结风险，在CPU上检查正确/错误分布、错误类PR/ROC、错误接受—正确误拒—覆盖关系，并按视频/real/sim对照score。该检查尚未执行，TRAIN拟合内与VAL开发/校准身份须披露。

若评分能区分而工作点不合适，下一版本才预先固定正确性目标及两类误判代价，不再以“尺寸/方向必须95%接受”替代判断正确。若评分不能区分，改门限或校准概率不足以解决；再限制到尺寸的二维图像范围与原框显式一致性实验。方向需实际参考足够准再做低维周期一致性，不能直接重训已关闭的共享结构/高维拼接头。

中心TRAIN只有1个真实错误，当前不把默认保留写成已解决中心正确性判别。不能通过全部拒绝或总准确率指标掩盖错误检出/正确误拒代价。新训练/门限/设计仅在TRAIN/VAL开发；当前TEST已多次暴露，只报告冻结方案，不重选权重、前端或规则，不称未接触独立确认。

此处为下一步建议，未修改代码或启动新优化实验。

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
