# Base V3 + V5.1 归档交接状态

更新时间：2026-09-15

## 当前版本

- 诊断代码：`base_v3_obb_hybrid_fixed_test_diagnostic_v51_v31`
- 统计修订：`v31_r1_absolute_tolerance`
- 比较容差：`1e-12`
- 归档入口：`base_v3_obb_focused_paper_result_archive_v31_r2`
- 统一模型与报告入口：`base_v3_obb_focused_paper_pipeline_v1`
- 统一入口配置：`crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json`

## 文件与 SHA256

- 历史诊断（保留）：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_obb_hybrid_policy_diagnostic_v51_v31.json`；`753b6460c192c2a4367b67cb020951d40d1adc383c7ea01f0bfcdbfc35402f38`
- 逐帧回放诊断：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_obb_hybrid_policy_diagnostic_v51_v31_r1_replay.json`；`d09d650e499da7482140cfbb810555b84ac232945bea4fb6811d801533d777f2`
- 历史差值重分类诊断：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_obb_hybrid_policy_diagnostic_v51_v31_r1_reclassified.json`；`3f30f5f720e8a3036dae8e3116f453f9581c5cad86af96db9168efdf52fa0ae7`
- 完整归档 JSON：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_focused_paper_result_archive_v31_r2.json`；`f5a2cb1907cf9dd90d8cf88bcc5a3f04330d3e195246f2ab375ee7542415eb82`
- 真实在线 V3：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_true_online_paper_metrics_v3.json`；`95a1671b9f53d1258e2db21053c14ecaea6bc22a08973dfc560d13fa762301cf`
- 模型流程总报告 JSON：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_focused_paper_model_pipeline_v1.json`；`28b99162598f9a0b6922ec0ab0e17888291e7d1c7d17b544c11e17e07f607530`
- 模型流程总报告 Markdown：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_focused_paper_model_pipeline_v1.md`；`96166c3c7806e345a50f5d88fc9405c37b27f31a86f4631688d4e1060973f394`
- 可靠性专项报告 JSON：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_focused_paper_reliability_v1.json`；`52ad9bfea5826096a4fc344f3a48574084003ebafebb0b85f4184c013e416884`
- 可靠性专项报告 Markdown：`work_dirs/base_v3_obb_reliability_baseline_v1/fixed_test_focused_paper_reliability_v1.md`；`acd7ba9f7d76c166ae6461c762c9e0d4e87ffca75f344480a48fbd24d4c557a1`
- 状态展示清单：`work_dirs/base_v3_obb_reliability_baseline_v1/focused_paper_visualization_v1/visualization_manifest_v1.json`；`b14e939cc1da8a4e35f8fc7c0b29243fc361fe1d71b02f9a832d39023bc14525`

## 已修复

- 输入契约协议与输出报告协议分离。
- 同覆盖率严格胜出和平局使用绝对容差 `1e-12`。
- 新归档绑定在线 V3 与历史诊断 SHA256，不覆盖旧结果。
- 真实在线性能与冻结 V5.1 诊断分开归档。
- 逐帧回放诊断与历史差值重分类诊断使用不同文件名，避免安全写入冲突。
- 统一入口支持 `validate`、`report`、`visualize` 和 `full` 四种模式；`full` 模式严格按图像、在线观测、后验 GT 评估和报告生成的顺序执行。
- 总报告固定包含完整 OBB 9 行、分量指标 27 行、锚来源 6 行及已有自定义诊断。
- 可靠性专项报告单列分量有效性、部分分量、联合正确性、同覆盖率尺度比较、角度保持和连续缺失统计，并解释 `valid` 不等于 GT 正确。
- 状态展示按不同分量状态组合的首次出现帧自动选取，不使用 GT 或误差排序；灰色框表示 Base V3，绿色框表示完整 V5.1 OBB，黄色点表示完整框不可用时仍有效的中心。
- `full` 模式从本次在线输出和后验误差重新计算可靠性诊断，不混用旧诊断 SHA；旧诊断仅供冻结 `report` 模式复现。
- 已删除 74 个退出主线的配置、75 个对应的实验/审计工具和 54 个失效或已退出范围的专项测试；清理范围包括检测方法比较、失败实验、source gate、seq11、S7/DINO 消融、dual-tower、深度及其派生审计。保留统一入口、Base V3/V5.1 正式在线与归档链、最终模型配置、训练/测试公共入口和冻结 DINO checkpoint 所需的公共工具。V5.1 冻结策略仍调用 V5 的尺度风险函数，因此保留 `base_v3_obb_heterogeneous_policy_development_v5.py` 作为运行依赖。

## 已验证

清理后的全部顶层项目测试：365 passed；统一入口自身回归测试为 8 passed。
诊断入口已从下载的冻结 V5.1 原始逐帧评估生成回放报告；归档入口已用真实下载报告和契约生成本地 JSON/Markdown。
统一入口 `validate` 和 `report` 模式已在本地真实绑定文件上执行；四份正式输出已经生成并记录 SHA256。
`visualize` 模式已生成 6 种实际出现的分量状态组合示例，清单和示例图经过读取与目视检查。

## 未解决/待服务器执行

- 两份报告的角度平均误差差异尚未完成逐帧核对。
- 服务器源码、契约和 Python 3.8 环境一致性尚未核验。
- 本地缺少服务器上的 `fixed_target_anchor_fallback_attribution.json` 和历史 `conservative_takeover_fusion_audit.json`，因此本地未执行需要完整模型依赖和这些重建输入的 `full` 模式。
- 未知序列泛化、物理状态和标准部署实时性仍不支持。

## 证据边界

固定 TEST 已暴露；不训练、不推理、不调参。V5.1 不是整体最优方法，部分分量可用性必须与完整观测准确性和覆盖率并列报告。

## 下载的 unified_full_run_v1 核验与自定义指标补充

- 已读取 `work_dirs/base_v3_obb_reliability_baseline_v1/unified_full_run_v1` 的 9 份服务器产物，全部 SHA256 与用户提供值一致；17 处指向这些文件的嵌套输入绑定一致。CSV 为 8928 行（992 帧 × 3 方法 × 3 分量）。原始产物未修改。
- 本次在线 JSON 和观测 CSV 与已有历史在线文件逐字节相同。这是同一固定 TEST 输出的再现，不构成新的独立测试集。
- 原统一报告遗漏了自定义指标。本轮新增 `crane_project/tools/analyze_unified_full_run_v1.py`，使用绑定的服务器逐帧误差/RIoU 和本次在线框，补算协议 2 的 R_center、A-RMSE、DFR、ACI、TDR、MCML、MRF；DEP 无物理真值，未计算。
- 完整补充 JSON：`unified_full_run_v1/verified_custom_metrics_v1.json`，SHA256 `ac2890a1d857bc49f87ec2d1bd86968d7eac395524a4e4776f593eb770520616`。
- 完整补充 Markdown：`unified_full_run_v1/verified_custom_metrics_v1.md`，SHA256 `aa62c4a848cd993f2d301e1eba2626512d5d749a809d6df62e00345f0d5afa06`。包含自定义指标、整体表和独立可靠性章节；可靠性原始独立 JSON/Markdown 继续保留。
- 未解决：本地 GT 重算与服务器逐帧误差最大差分别为中心 0.0444445893 px、尺度 1.03489105e-6、角度 0.0522689819°、RIoU 0.00133404516。原因未确认；未用本地重算替换服务器值。补充 JSON 记录了本地标注 SHA256 和差异帧。
- 实际验证：`tests/test_analyze_unified_full_run_v1.py` 与 `tests/test_eval_crane_offline_records.py` 合计 7 passed；合成序列验证冻结误差汇总与原几何评估器一致，覆盖缺测、帧号间断、顺序变化和非法记录。本地真实补充入口执行成功，并对 real/sim 的自定义平均 RIoU 与服务器完整框表交叉核验。
- 本轮没有运行模型推理、训练或服务器 Python 3.8 验证。统一 full 入口尚未自动调用本次补充脚本；补充入口专用于这批 SHA 绑定产物。在仓库根目录执行 `PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 python -m crane_project.tools.analyze_unified_full_run_v1` 可在当前本地标注环境复现。
- 使用范围：可作为冻结实现的固定 TEST 结果；不能宣称整篇论文实验已齐备、V5.1 整体优于置信度筛选或未知序列泛化成立。运行时间包含初始化、图像 I/O、检测与报告，不等于纯推理或可靠性模块单独开销。

## 统一入口分阶段修订

- 统一入口新增 `infer` 和 `evaluate`，形成 `infer → evaluate → report` 三阶段；`full` 依次复用三阶段。`infer` 输出在线 JSON、无 GT 观测 CSV 和 inference receipt；`evaluate` 只在在线输出完成后连接 GT。
- `report`、`validate` 和 `visualize` 可通过 `--input-dir` 读取指定运行目录，不再只能使用配置中冻结的历史报告。
- 自定义 R_center、mean_RIoU、A-RMSE、DFR、ACI、TDR、MCML、MRF 已进入统一模型 JSON/Markdown；可靠性报告继续单独输出。DEP 因没有物理深度真值保持不可用。
- 新评估会保存 finalization 运行时契约、paper-metrics 运行时契约、诊断输入和诊断运行时契约。每份契约绑定本次实际在线 JSON、CSV 或上游报告 SHA256，模板契约不在内存外被改写。
- 在线观测 CSV 改为精确写入：相同内容可重复验证，目标存在不同内容时拒绝覆盖。
- 修正运行时间语义：旧 V1 标签写成包含 reporting，但计时代码实际只包围在线推理子进程。新 inference receipt 使用 `model_init_detector_image_io_and_online_json_generation`，不把 GT 后验评估或最终报告写入计入该时间。
- 同一次运行生成的论文角度指标和诊断角度指标明确标记为同一逐帧来源；历史不同来源仍保留“待逐帧解释”的限制。
- 实际验证：统一入口、自定义指标和离线记录测试合计 `20 passed`；相关在线 pipeline、finalization、paper metrics、诊断、论文 finalization 和 V5.1 eval 回归合计 `45 passed`；全部顶层项目测试 `372 passed`。生成目录 `validate --input-dir` 成功；动态 `report --input-dir` 成功生成四份报告并包含 `real/R_center(%)`。
- `pytest tests` 的全树收集在本地 Python 3.13 环境出现 15 个依赖导入错误，均来自上游 MMRotate 数据、模型和工具测试缺少 `mmcv`/`mmdet`，没有进入测试执行。这不是本轮代码断言失败；应在服务器 `mmrotljj` 环境补跑所需上游测试。
- 本地未执行 CUDA 推理和真实 `evaluate`，因为缺少服务器上的 attribution 与 all-lane audit 两份重建输入。服务器应先同步本节列出的源码、配置、测试与文档，再执行回归和新的分阶段或 `full` 命令。
