# SymEOOD＋尺度增强B＋simple v1：小论文模型与实验完整流程

更新：2026-10-03，依据服务器最新TEST回传完成替换。文件名沿用旧路径，仅为兼容链接；正文模型、数据、指标和结果均为当前版本。旧正文单独见[归档](../archive/20261003_reliability_replaced_by_simple_v1/readable/docs/obb/base_v3_v51_focused_paper_complete_pipeline_20260915.md)，不再作为当前模型或成绩引用。

项目目录：`/Users/mac/Documents/paper/symEOOD`；服务器目录：`/media/omnisky/personal_files/ljj/symEOOD`。

## 1. 研究范围与当前模型链

```text
当前港口RGB图像
  → SymEOOD＋等比例尺度增强B（VAL选epoch24）
  → 原图旋转框与真实检测score
  → simple v1：中心、尺寸、方向三个独立使用标志
  → 离线接GT进行分量正确性、覆盖及完整OBB评价
```

当前模型不使用DINO、候选融合、Base V3/V5.1、历史保持或缺测预测。连续状态、物理摆角、深度/空间风险及报警接口留给大论文。

模型、接口、缓存报告闭合已完成；可靠性增益尚未成立。完整流程不等于投稿创新或全部公平检测实验已齐备。检测侧EOOD/SymEOOD及匹配尺度增强对照仍维护在[检测主记录](../detection/港口新数据集与EOOD_SymEOOD尺度增强实验总记录.md)，本次不更换固定B。

## 2. 模型身份与方法

B配置：`crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py`；checkpoint：`work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth`，SHA `8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23`。

simple policy：`work_dirs/port_simple_reliability_v1_fit/policy.json`，SHA `bbc3075fdb2f6a9f817cffa645b82aa43b4be9ca95ede4f929c23f02e784d7a8`。本次保留固定版本，不由TEST重选。

中心标志保留每个有效B输出，不另训练中心正确性预测器。尺寸、方向各为独立线性风险模型，输入只有当前框的score logit、相对几何尺寸和长宽比；共8个拟合系数。原图长短边只在风险计算中规范化，导出的原宽高、角度、中心、score及输出数量保持不变。

TRAIN2558帧（real1810、sim748）用于拟合/标准化；VAL887帧（real375、sim512）用于此前固定的95% pooled覆盖规则产生单一全局工作门限，之后冻结。real原生轴线转换OBB和sim Webots生成OBB提供监督；sim没有轴线标注，不要求补轴线。GT不进入在线API。尺寸/方向平衡类sigmoid是风险评分，不是校准错误概率。

三特征为`logit(score)`、`.5*(log L+log S-log W-log H)`、`log(L/S)`，L/S为预测规范长短边，W/H为原图尺寸。固定正负类各总权重0.5、L2=0.1、float64 Newton最多100次/梯度容差1e-8；本次TEST不再拟合或调整这些参数。

冻结门限：simple尺寸风险≤0.4716593377139233、方向≤0.516876854517756；score-only两分量风险`1-score`≤0.7043237686157227。95%是原VAL校准目标，不能保证TEST每域/每分量95%接受。所有有效B中心都保留。

## 3. 数据、在线输入及评价

数据根目录：`crane_project/data/crane_grab_port_day2night_v1`。TRAIN2558＋VAL887＋TEST1440=4885帧；按视频职责保留原划分，不重新随机分割帧。

TEST1440帧：real_seq03 200、real_seq04 668、sim_seq09 572。缺失9帧全部保留，原输出1431帧。本版TEST全部1440帧方向可评。

中心正确为原图欧氏误差<15px；尺寸正确为规范化长短边最大相对误差≤10%；方向正确为长边π周期角误差≤3°。GT aspect≥1.2仅用于离线方向评价，不用于在线接受决策。中心命中率仅以输出帧为分母，另报输出覆盖和全帧中心正确覆盖。

`raw`保留所有有效B分量；`score_only`用冻结VAL门限筛选尺寸/方向、中心仍全部保留；`simple`用两组冻结线性风险与VAL门限分别判尺寸/方向、中心仍全部保留。无B输出时三个标志均false。GT只在评分后定义正确/错误，不用于在线判断。

同接受数score只作离线排序诊断：匹配simple数量，不能把它称为部署门限或由TEST校准的score。固定score门限与同数score对照是两种不同职责。

原始输出保持`(cx,cy,w,h,angle,score)`关联，不将规范化角度替换导出的原角。在线只用当前原框与原图大小，输出`center_accepted/size_accepted/angle_accepted`；三个标志都是决策，不是读取GT生成标签。缺失帧三个false且保留评价分母。等比例变换、坐标还原和既有深度接口保持；深度精度未因此验证。

## 4. 最新服务器TEST结果

### 4.1 中心覆盖

| 域 | 总帧/输出/正确中心 | 输出覆盖率 | 输出帧中心命中率 | 全帧中心正确覆盖 |
|---|---:|---:|---:|---:|
| real | 868/859/858 | 98.9631% | 99.8836% | 98.8479% |
| sim | 572/572/572 | 100.0000% | 100.0000% | 100.0000% |

### 4.2 固定门限方法

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

### 4.3 同接受数参考与完整OBB

| 域/分量 | simple接受 | simple错误接受 | 同数score错误接受 | simple正确误拒 |
|---|---:|---:|---:|---:|
| real/size | 796 | 214 | 213 | 28 |
| real/angle | 742 | 132 | 137 | 73 |
| sim/size | 572 | 26 | 26 | 0 |
| sim/angle | 572 | 52 | 52 | 0 |

| 方法 | 完整OBB接受/1440 | 完整OBB覆盖 | 共同正确/1440 | 全帧共同正确覆盖 |
|---|---:|---:|---:|---:|
| raw | 1431/1440 | 99.3750% | 1002/1440 | 69.5833% |
| score_only | 1380/1440 | 95.8333% | 986/1440 | 68.4722% |
| simple | 1311/1440 | 91.0417% | 961/1440 | 66.7361% |

real最长标志不可用段中心4、尺寸16、方向17帧；sim均0。这是拒绝/漏检统计，不是历史补全或物理状态稳定性证明。

## 5. 结果分析与论文主张

**事实：** real尺寸同数错误214对213，未优于score；方向132对137有5个错误的局部减少，但误拒73个正确方向。sim全部接受，26个尺寸错误和52个方向错误均未检出；其较高总准确率来自原B多数正确，不能证明判别有效。real错误检出率尺寸14.0562%、方向25%，正确观测保留率95.4098%/89.3119%。

real状态误判总数尺寸214+28=242/859、方向132+73=205/859，状态准确率71.8277%/76.1350%，均低于固定score的72.8754%/78.5797%。本文主结果以本次服务器TEST为准。

逐视频同数对照：seq03尺寸71对75、方向36对43；seq04尺寸143对139、方向96对95，收益不一致。全局尺寸240对242、方向184对194的局部改善不能证明各域/各视频稳定优势。拒绝降低了完整OBB和正确分量的覆盖，不改善原B定位/几何。

**判断：** 本版工程流程与报告闭合，可靠性提升的主张尚未成立；适合作为简单基线和负结果，不宜承担论文核心新方法的成功结论。较高sim状态准确率不抵消零错误检出的事实。

**静态事实：** 三个风险特征不包含图像边界、参考轴或预测角度值；只改变预测角度而保持score和长短边时，风险不变。因此当前方向评分依赖统计相关性，没有直接方向一致性证据。这是输入机制限制，不是已证明的跨视频退化唯一根因。

**推断，未证实根因：** 当前三特征没有直接图像范围/方向证据，TRAIN与新视频的错误分布可能不同，工作点也可能偏宽松；这些都需在TRAIN/VAL检查，不根据本TEST逐视频反推新规则。

论文可写：固定高覆盖检测产生不同质量的分量；建立独立使用接口，量化错误接受、正确误拒与覆盖代价。当前simple作为基线，失败和局部收益同时呈现。

论文暂不能写：三个分量均被可靠判准、跨视频稳定或显著优于score、几何精度/深度精度改善、物理安全报警或未知域验证。VAL已用于开发和工作点，TEST已多次暴露；本次仅为冻结方案在既有划分上的报告。

## 6. 当前产物与完成身份

服务器包：`port_simple_reliability_v1_test_review_20261003_205844.tar.gz`，SHA `35b7d585cd0c94e4deb61238a6ecdac7850284e6fa9942fd2815b37fdc1e52cd`。所有产物的完成SHA、1440帧身份、12960标志及72混淆表已核对，通过`RETURNED_SERVER_SIMPLE_V1_TEST_REVIEW_PASS`；该数据是保存的服务器原B/GT数值，不是本轮新图像前向。

本地最新回传目录：`work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844`；核心文件为`test_compare.json`、`test_decisions.jsonl`、`component_metrics.csv`、`paper_flow_and_results.md`、`completion.json`和外层`review.json`。

最新报告SHA `1f19521eecbf7a70d8cc0fb808215a4919590ef4f82189667de4aad157879440`；逐帧SHA `174e5ab1632c42f9b62d756029b21682196900c26940ef07058dfb07d7160430`。CSV和论文流程MD与本地收尾逐字节一致；所有统计一致，风险末位浮点差异不影响标志。当前以服务器原始文件为报告依据。

代码：`run_port_simple_reliability_v1.py`、`port_simple_component_reliability_v1.py`、`eval_port_simple_reliability_v1_test.py`；完整协议/来源见[当前交接](../reliability_handoff_20261003.md)。不再使用旧完整流程执行脚本。

## 7. 复现入口（已完成，不必重跑）

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

成功状态：`FIXED_SIMPLE_V1_TEST_CLOSURE_COMPLETE_REVIEW_REQUIRED`。当前服务器check/run均已成功，不需要GPU或新训练。工程完成与方法增量分别判断。

## 8. 下一阶段

本版policy/门限/B及新TEST报告保留冻结，当前流程收尾完成。下一项先复用既有TRAIN/VAL原框、GT与冻结风险，在CPU上检查正确/错误分布、错误类PR/ROC、错误接受—正确误拒—覆盖关系，并按视频/real/sim对照score。该检查尚未执行，TRAIN拟合内与VAL开发/校准身份须披露。

若评分能区分而工作点不合适，下一版本才预先固定正确性目标及两类误判代价，不再以“尺寸/方向必须95%接受”替代判断正确。若评分不能区分，改门限或校准概率不足以解决；再限制到尺寸的二维图像范围与原框显式一致性实验。方向需实际参考足够准再做低维周期一致性，不能直接重训已关闭的共享结构/高维拼接头。

中心TRAIN只有1个真实错误，当前不把默认保留写成已解决中心正确性判别。不能通过全部拒绝或总准确率指标掩盖错误检出/正确误拒代价。新训练/门限/设计仅在TRAIN/VAL开发；当前TEST已多次暴露，只报告冻结方案，不重选权重、前端或规则，不称未接触独立确认。

此处为下一步建议，未修改代码或启动新优化实验。
