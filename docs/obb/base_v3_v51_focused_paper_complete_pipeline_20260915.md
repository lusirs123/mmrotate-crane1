# B24＋正式midpoint23＋simple三标志：固定基线流程记录

更新：2026-10-04。用户已将本版固定TEST流程确定为后续可靠性实验基线。文件名兼容原链接，本文是项目记录，不是实际小论文或大论文稿。旧B＋simple结果按历史身份保留[可靠性交接第3节](../reliability_handoff_20261003.md)，执行状态与后续版本集中在该交接第18/19节；更早模型见[原归档](../archive/20261003_reliability_replaced_by_simple_v1/readable/docs/obb/base_v3_v51_focused_paper_complete_pipeline_20260915.md)。

项目`/Users/mac/Documents/paper/symEOOD`；服务器`/media/omnisky/personal_files/ljj/symEOOD`。

## 1. 固定模型链与对照角色

```text
当前港口RGB图像
  → 冻结SymEOOD＋等比例尺度增强B（epoch24）
  → 冻结正式midpoint（原完整VAL规则选head_epoch_23）
  → 最终原图旋转框与原score
  → 现有simple中心、尺寸、方向三个独立使用标志
  → 离线GT评价正确性、误判、覆盖及完整OBB
```

固定流程、接口及1440帧TEST报告已完成，性能取舍不妨碍其作为对照。后续可靠性候选首先保持同一B/midpoint及输出，仅改变质量证据/判断，原simple始终作为冻结参考。尺寸模板、旧ROI/structure、DINO、Base V3/V5.1、历史保持均不进入本基线。连续状态、物理摆角、深度/空间风险及报警留给大论文。

## 2. 冻结身份与方法

B配置`crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py`，权重`work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth`，SHA `8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23`。

正式midpoint权重`work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/head_epoch_23.pth`，SHA `2c4c5ae9e071cbdb13fb722f4b2d3986861c62bd60c53568c72fb9b02d662f0b`，原selection SHA `5e4b24eb15f8fcc925c95445d1f6ad3c7fa643b350267b3647710a0d14024ec6`。

simple policy为`work_dirs/port_midpoint_reliability_v1_fit/policy.json`，SHA `1adac499369bade5ef1f79495260002697d2b698c80ebf5424d3d77438fae4af`；已经按新前端TRAIN/VAL拟合，不用旧B policy代替。

中心标志保留每个有效最终midpoint输出，不另训练中心正确性预测器。尺寸、方向各为独立线性风险模型，输入是最终框的score logit、相对几何尺寸和长宽比，共8个拟合系数。风险计算规范长短边，但原图最终中心、宽高—角度关联、score和输出数量不被可靠性改变；尺寸/方向拒绝不删除中心。

TRAIN2558帧（real1810、sim748）用于拟合/标准化，VAL887帧（real375、sim512）按原95% pooled覆盖规则产生全局工作门限，现已冻结。real监督来自原生轴线转换的OBB，sim来自Webots生成OBB，sim无需轴线标注。GT只用于训练和离线评价，不进入在线判断。平衡类sigmoid是风险评分，不能直接称为校准错误概率。

三特征为`logit(score)`、`.5*(log L+log S-log W-log H)`、`log(L/S)`；L/S是最终框规范长短边，W/H是原图宽高。拟合规范为正负类各总权重0.5、L2=0.1、float64 Newton最多100步/梯度容差1e-8。当前使用midpoint前端的已审policy，旧B policy仅作历史/配对对照。

冻结门限：simple尺寸风险≤0.39773387607871297、方向≤0.4143115365368443；score-only两分量风险`1-score`≤0.7043236196041107。95%是原pooled VAL目标，不保证TEST每域95%接受。所有有效最终框中心均保留。

## 3. 数据与评价

数据根目录`crane_project/data/crane_grab_port_day2night_v1`，TRAIN2558＋VAL887＋TEST1440＝4885帧，沿用原视频划分。TEST为real_seq03 200、real_seq04 668、sim_seq09 572；9个缺输出保留，最终输出1431帧，方向GT资格1440/1440。

中心正确为原图误差<15px，尺寸正确为规范长短边最大相对误差≤10%，方向正确为长边π周期误差≤3°。GT aspect≥1.2仅用于离线方向资格。中心命中率只统计输出帧，另报输出覆盖及全帧正确中心覆盖。

`raw`保留全部有效最终框分量；`score_only`用冻结VAL score门限判断尺寸/方向；`simple`分别使用冻结线性风险和门限。三方法中心都保留，无输出时三个标志均false。状态误判=错误接受＋正确误拒，状态准确率只以有输出的可评帧为分母；缺输出另报。

同接受数score是离线诊断，按与simple相同数量比较，不生成TEST部署门限。在线仅接收当前最终框/score和原图宽高，不接收GT、域/视频身份或历史未来。保持等比例变换、原图坐标还原和既有深度接口；固定接口不额外验证深度精度。

## 4. 当前固定TEST主表

### 4.1 中心

| 域 | 总帧/输出/正确中心 | 输出覆盖率 | 输出帧中心命中率 | 全帧中心正确覆盖 |
|---|---:|---:|---:|---:|
| real | 868/859/858 | 98.9631% | 99.8836% | 98.8479% |
| sim | 572/572/572 | 100.0000% | 100.0000% | 100.0000% |

### 4.2 固定门限三方法

| 域/分量 | 方法 | 接受/总帧 | 错误接受 | 正确误拒 | 接受后正确率 | 输出帧状态准确率 |
|---|---|---:|---:|---:|---:|---:|
| real/size | raw | 859/868 | 379 | 0 | 55.8789% | 55.8789% |
| real/size | score_only | 819/868 | 349 | 10 | 57.3871% | 58.2072% |
| real/size | simple | 737/868 | 289 | 32 | 60.7870% | 62.6310% |
| real/angle | raw | 859/868 | 184 | 0 | 78.5797% | 78.5797% |
| real/angle | score_only | 819/868 | 163 | 19 | 80.0977% | 78.8126% |
| real/angle | simple | 781/868 | 145 | 39 | 81.4341% | 78.5797% |
| sim/size | raw | 572/572 | 58 | 0 | 89.8601% | 89.8601% |
| sim/size | score_only | 561/572 | 56 | 9 | 90.0178% | 88.6364% |
| sim/size | simple | 572/572 | 58 | 0 | 89.8601% | 89.8601% |
| sim/angle | raw | 572/572 | 6 | 0 | 98.9510% | 98.9510% |
| sim/angle | score_only | 561/572 | 6 | 11 | 98.9305% | 97.0280% |
| sim/angle | simple | 572/572 | 6 | 0 | 98.9510% | 98.9510% |

### 4.3 同数score诊断与完整OBB

| 域/分量 | simple接受 | simple错误接受 | 同数score错误接受 | simple正确误拒 |
|---|---:|---:|---:|---:|
| real/size | 737 | 289 | 305 | 32 |
| real/angle | 781 | 145 | 155 | 39 |
| sim/size | 572 | 58 | 58 | 0 |
| sim/angle | 572 | 6 | 6 | 0 |

| 方法 | 完整OBB接受/1440 | 完整OBB覆盖 | 共同正确/1440 | 全帧共同正确覆盖 |
|---|---:|---:|---:|---:|
| raw | 1431/1440 | 99.3750% | 922/1440 | 64.0278% |
| score_only | 1380/1440 | 95.8333% | 907/1440 | 62.9861% |
| simple | 1307/1440 | 90.7639% | 899/1440 | 62.4306% |

real最长标志不可用段中心4、尺寸21、方向16帧；sim均0。它们是漏检/拒绝统计，不能作为物理状态稳定性证明。

## 5. 结果身份与解释

本基线已完成固定流程和TEST报告，用户明确将其作为后续比较起点，不要求先达到最优性能或优于score。现有取舍照实保留，后续新方法在同一前端和评价条件下比较。

real尺寸状态误判289＋32＝321/859、状态准确率62.6310%；方向145＋39＝184/859、78.5797%。sim尺寸58/572、方向6/572，全部接受、错误检出率0；其高准确率不能归功于筛选。real同接受数尺寸/方向比score各少接受16/10坏；逐视频尺寸seq03为49对67、seq04为240对241，方向43对49、102对105。各组独立匹配数量，不能将逐视频score计数相加当成域级计数。这些是基线描述，不是新方法显著提升主张。

midpoint改变最终框，尺寸标签在判断前已变化：同次B→midpoint的raw错误real尺寸249→379、sim26→58，real方向176→184、sim52→6。simple绝不改框，尺寸误差不能靠拒绝判断修复；新旧迁移同时改变几何及risk参数，不能把尺寸状态准确率下降全部归因于判别器变差，也不能说三个分量都恶化。更高RIoU/更准定位与严格10%尺寸通过率下降可以同时发生。

当前三特征没有直接图像边界/参考方向证据，只改变角度而保持score和长短边时方向风险不变；它依赖相关性。该机制限制与工作点/跨视频泛化均需按TRAIN/VAL研究，不是唯一根因已经确认。

本表是当前基线的事实起点，不把“已固定”为“性能创新成功”。稳定/显著增量、深度精度、物理安全报警与未知域确认需各自证据。实际大小论文稿暂不改。

## 6. 当前产物与完成身份

服务器包`port_midpoint_simple_reliability_v1_test_review_20261004_190008.tar.gz`，SHA `5c93f6fdc250aa38af2480b9d0102c5ed6e2a4d617f6e9ff390eef37c9d2f0d1`。check终态`FIXED_MIDPOINT_SIMPLE_TEST_INPUTS_PASS`；run终态`FIXED_MIDPOINT_SIMPLE_TEST_COMPLETE_REVIEW_REQUIRED`。8个文件、6个完成清单产物SHA、当前源码/冻结输入身份、全部1440行和三标志/统计已复核，具体见可靠性交接第18节；不是本地新GPU前向。

当前主报告SHA `5dbf8fd1f6623cfa62f1c7b7ea94360dc2049a1ef2f3448d6fd8f790fa53053e`，逐帧SHA `f625d1a5e351a0717e3343876ba456369102972005938888c9cf837b1c3226b2`。仅在内存读取回传包，不创建本地新结果目录。服务器结果目录`work_dirs/port_midpoint_simple_reliability_v1_test_cached_v1`，主要文件`test_compare.json`、`test_decisions.jsonl`、`component_metrics.csv`、`test_summary.md`、`input_check.json`、`completion.json`。

入口`crane_project/tools/eval_port_midpoint_simple_reliability_v1_test.py`，固定协议与来源清单同名前缀。复用正式midpoint TEST的同次最终/B框，旧缓存只补图像宽高与数据身份，旧ROI/structure分数不参与；paired_B_old_simple是历史迁移对照，不是本轮可靠性优化版本。

## 7. 复现入口（已完成，无需重跑）

```bash
cd /media/omnisky/personal_files/ljj/symEOOD

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_midpoint_simple_reliability_v1_test.py \
  --mode check --out-dir work_dirs/port_midpoint_simple_reliability_v1_test_check

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python crane_project/tools/eval_port_midpoint_simple_reliability_v1_test.py \
  --mode run --out-dir work_dirs/port_midpoint_simple_reliability_v1_test_cached_v1
```

上述阶段已成功；以后确需复现时使用新输出目录，不覆盖本基线。默认依赖原midpoint fit目录、`work_dirs/port_geometry_midpoint_formal_v1_test_eval`、原B policy及原尺寸元数据缓存；完整默认参数见可靠性交接第17节。CPU评分不加载模型/图像，不需要GPU、训练或再次检测。

## 8. 下一阶段分工

比较基线收尾已经完成，原B24、正式midpoint23、simple参数/门限及结果保持不动；后续另立可靠性候选版本，在同一最终框上比较错误接受、正确误拒、错误检出和正确观测覆盖。此前可分性、参考训练、模板读取及全量VAL检查都已完成，执行详情见可靠性交接第8—18节，不重复写成尚未开展或要求重跑固定TEST。

最推荐下一候选限制到尺寸可靠性，沿已有TRAIN/VAL证据增强图像尺寸参考的核心范围/长短边监督；冻结检测和midpoint、模板读取及坐标链，先固定公式/系数/预算/支路留出角色，做有限TRAIN对照，再固定全量VAL评价。参考自己更准是否带来错误辨识收益仍待验证，不只看loss下降或多训几轮。新工作点如需改变，应先规定TRAIN/VAL目标及误判代价，原95%基线仍保留。方向结构分支暂不并行重启。

要修复最终框尺寸，属于另一几何窗口的TRAIN/VAL工作；不能用本可靠性分支改框、统一放大或按TEST视频回退。即使以后另选几何前端，也保留本固定基线，分开报告前端变化和可靠性变化，不自动覆盖参数/成绩。TEST已多次暴露，本版只报告冻结流程，后续不据其选权、结构或门限。中心默认保留不等于已判准，三个使用标志不等于物理安全判定；连续状态、摆角与风险报警留给大论文。当前未启动新优化实验或修改实际大小论文稿。

## 9. 旧B＋simple来源（历史保留，不是当前主基线）

历史B policy为`work_dirs/port_simple_reliability_v1_fit/policy.json`，SHA `bbc3075fdb2f6a9f817cffa645b82aa43b4be9ca95ede4f929c23f02e784d7a8`；原simple尺寸/方向风险门限分别0.4716593377139233/0.516876854517756，score-only风险门限0.7043237686157227。这些旧参数不用于当前midpoint主基线。

历史B＋simple服务器包：`port_simple_reliability_v1_test_review_20261003_205844.tar.gz`，SHA `35b7d585cd0c94e4deb61238a6ecdac7850284e6fa9942fd2815b37fdc1e52cd`。所有产物的完成SHA、1440帧身份、12960标志及72混淆表已核对，通过`RETURNED_SERVER_SIMPLE_V1_TEST_REVIEW_PASS`；该数据是保存的服务器原B/GT数值，不是本轮新图像前向。

历史B回传目录：`work_dirs/port_simple_reliability_v1_test_server_review_20261003_205844`；核心文件为`test_compare.json`、`test_decisions.jsonl`、`component_metrics.csv`、`paper_flow_and_results.md`、`completion.json`和外层`review.json`。

历史B报告SHA `1f19521eecbf7a70d8cc0fb808215a4919590ef4f82189667de4aad157879440`；逐帧SHA `174e5ab1632c42f9b62d756029b21682196900c26940ef07058dfb07d7160430`。CSV和论文流程MD与本地收尾逐字节一致；所有统计一致，风险末位浮点差异不影响标志。这些文件仍按历史B身份保留，不作为当前midpoint主表。

代码：`run_port_simple_reliability_v1.py`、`port_simple_component_reliability_v1.py`、`eval_port_simple_reliability_v1_test.py`；完整协议/来源见[当前交接](../reliability_handoff_20261003.md)。它们是历史B入口，当前基线入口见第7节。
