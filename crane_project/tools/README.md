# crane_project/tools：用途导航与清理记录

整理日期：2026-10-28。原Git跟踪文件615份，本次删除49份（{'.py': 45, '.json': 3, '.sh': 1}），其余文件内容和文件名保持。额外删除可再生成的`__pycache__`／`.pyc`及`.DS_Store`，不另存一套旧代码。

## 当前优先入口

| 用途 | 文件 |
| --- | --- |
| 新110原预算训练与结果汇总 | `run_symeood_110_retrain_v1.py`、`analyze_symeood_110_retrain_v1.py` |
| 新110 midpoint | `train_symeood_110_midpoint_v1.py`、`eval_symeood_110_midpoint_ep21_test.py` |
| 原共享LVSR共同阶段 | `run_symeood_110_lvsr_v1.py` |
| 小论文双分支LVSR | `train_symeood_110_lvsr_pair_v1.py`、`run_symeood_110_lvsr_pair_v1.py`、`eval_symeood_110_lvsr_pair_ep11_test_v1.py` |
| 大论文共享LVSR ep11 | `eval_symeood_110_lvsr_ep11_test_v1.py`，读取原fixed_b共同阶段ep11，不接C续训 |
| 通用评价／选版 | `eval_crane_offline.py`、`ckpt_sweep.py`及原AP入口 |
| 通用工程功能 | `train.py`、`test.py`、分布式／Slurm脚本、data转换、deployment和model_converters保持 |

这是文件导航，不是本轮运行命令或新实验授权。当前大小论文权重身份和性能见[论文总览](../../docs/detection/00_论文总览与阅读导航.md)。

## 清理边界

保留当前110→midpoint→共享／双分支LVSR的冻结来源，保留仍被项目代码、现有测试、完整Shell入口和相关协议调用的依赖。专项入口退出当前流程且不在这些闭包中时才移除。未仅凭文件日期或名称批量删除。

历史执行收据、旧协议和文档中的路径继续表示当时版本，不通过修改旧SHA伪装成仍可在当前工作区直接执行。需要重放某个旧研究时，从Git恢复该版本的脚本及其协议／配置依赖，或在另一个checkout运行该提交。当前保留列表不代表所有模块重新成为论文方法。

## 本次删除清单与恢复

删除前逐项核对原始字节，全部与提交`030395be620d743b64db3bb94bb9ae8d8eb05828`一致；Git历史没有删除或重写。清单中的SHA是该提交对应文件的完整SHA256。

| 删除文件（相对本目录） | 原因／范围 | 原始SHA256 |
| --- | --- | --- |
| `analyze_port_geometry_size_axis_balance_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `91fd44602cba09e779a4e029032e4c08c479fc7c9a6748024fa318351db57093` |
| `analyze_port_geometry_size_proposal_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `a615cf30814a67e7eb145e47167fc9a5dc400dc184c12e5e8dc1dbecc7035204` |
| `analyze_port_size_frozen_downstream_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `5744965b2ad9a9491de99e6be7da5ec1314eafd3fa0417b60603b0e0ee30fa26` |
| `analyze_port_size_local_vectors_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `3298722bb3be28df2adb0953663b7ddb48f2a39e8ac3ac3240743c02a693a386` |
| `analyze_port_size_native_train_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `671c0166795a92f8c7281ecc0dca7220f8aa223b80b30e4b2a1f1bbeb403608d` |
| `analyze_symeood_c_frozen_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `17b49953b47d58ad1be277dd753a4cc0febc4b8f94544dedc31abb264cda40d6` |
| `analyze_symeood_c_pair_postprocess_v2.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `91f1c682f38a2fcd88e2211b56eb401364a82c6d16332f552baebce930740469` |
| `analyze_symeood_mal_pair_v1.py` | 已结束的MAL／S_MAL专项入口或结果复核 | `891885cc77f38e757da303c790fb3bc5f9514ee465b71e9ee25481d264fb964b` |
| `analyze_symeood_mal_stop_v1.py` | 已结束的MAL／S_MAL专项入口或结果复核 | `7e3c871c4c59e1a5b658f6e9cba8fabe2185f4663c0f4b106e99d0d498418209` |
| `analyze_symeood_mal_val_only_v1.py` | 已结束的MAL／S_MAL专项入口或结果复核 | `5494c773b1edbf36fb2ee3d4e1f4aa767d8b5e100457d0bfb2fe8d5e3a3ad0a5` |
| `audit_historical_b_ablation111_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `0f6b20b0e0cff75239a5d9e5efe1629acbe590d88110c6e28bef8487238b9c6e` |
| `audit_port_size_auxiliary_conflict_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `f652ed554a7c9ef0431e704b040c3b3e28ea8aabbe73691084c69f0eb6e8c46a` |
| `audit_symeood_cross_entry_boundaries_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `be209a7efedcfdece9be0a8c7f8b6084343fd243be28679d9b58521be9ffb550` |
| `audit_symeood_cross_entry_history_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `58a02a289673abec80b07de4803525283f2e241ee4ad6a07afbd3fb2c26f080e` |
| `audit_symeood_cross_entry_identity_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `4b6bf7dec0391f97aea6a163f0a292e2789d71bebee6a5cfe088274db268797d` |
| `audit_symeood_cross_entry_serial_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `73d48cfc800290820c7270abe13e863407ba2cdfd2d9cf3150d77461a156782c` |
| `check_historical_b_loss_boundary_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `f896bb4575888584e21045e64497b390c575edeb784e97a30d87e9c6e57dae8c` |
| `check_port_oof_momentum_restart_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `51e32e70753ca5acfecc057d7269ba19050d6b2e8176360afeeb9a91b9a10c3f` |
| `check_port_oof_snapshot_detection_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `8b0d38bf81fdfb41941e41605a4c958b38813aa421b4ca7a184d6a93918f5800` |
| `check_port_oof_task_memory_v2.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `fd1b20b3e25e67e7d9756169858f15602fa4ac8f9f21e09ac36a1283612ebc93` |
| `check_symeood_usg_preflight_v1.py` | 已结束的USG专项准备／训练／评价入口 | `7cc3525349e9aca193fc2c8355b504e5d29f4dc988e1d60ed9d6f299fe945aa3` |
| `diagnose_port_oof_classification_stop_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `4038160813132c389f3003b4eef27a2fa2e908e87c1e76ebc97b31c77210d4cf` |
| `diagnose_port_oof_prevention_memory_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `b30be8d7b03e97b200f154fe3994b1edb5591e3661c43a862747fc12dd8d6dae` |
| `engineer_symeood_usg_v1.py` | 已结束的USG专项准备／训练／评价入口 | `15b36fe050dc87947fcd25171c2e3097cf95716f41b4e0cb496976887fb80686` |
| `evaluate_symeood_usg_v1.py` | 已结束的USG专项准备／训练／评价入口 | `804462aed92923646635cf0343a8be2c9cb6e5b09d53194c64deea793b140140` |
| `finalize_symeood_canonical_cpu_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `7893ca76c0e6042f5f754cc0f69cde84d256750af0cfd6aea01e65ae9432aa51` |
| `gate_symeood_c_pair_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `28894fc2d60b7e0f7febaa71ffb39641cb973aa5288c5b578aa9f9f0a15d8195` |
| `port_size_local_aux_v1_protocol.json` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `317b505caa15eb8dcc2465fb0c096dda904f89d0ff2726d892078d356efb1508` |
| `port_size_seven_replay_v1_protocol.json` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `4b95f395e362ef95fd9db375734c7aef221f0bd88a69b939b06fd41cbd474cbc` |
| `port_size_seven_replay_v1_sources.json` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `f96833dfc6f046cea5602225b1b23c4385ea0f066fa6e98381ba9332cd60cf4f` |
| `preflight_k1_dino_object_background_relation_v5.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `a6a6ad7b569a54cf37e228572d3aff9028ee7f5541ae5c65101a4ae3ea16c2af` |
| `preflight_symeood_usg_v1.py` | 已结束的USG专项准备／训练／评价入口 | `055057acee1b87b664ba39794af35d02c2f071bd7f94b1e19bd84cda5582d3cc` |
| `prepare_symeood_canonical_plan_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `e2324b54f7c9684884f102e3dfb13f93107381938c9120cd50b54b76b270467c` |
| `queue_symeood_cross_entry_review_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `c5fbebe9c1c8992e69b568206d13c9a6a9a38963d31aa2d452722f17c763f247` |
| `review_port_reliability_temporal_stack_test_boundaryfix1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `c3ff942aca68a0e55af2cc87c27632ea62f30e2073a6d04d6736e0eec5fbb8cc` |
| `run_port_oof_momentum_restart_prevention_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `ce15a3baf045f5afc213d9e7b4e3b970625edeb1b2bdf9b13f239e0d2c885b53` |
| `run_port_oof_task_budget_prevention_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `edd7ae973e179db70572b646debf6c8dfdac4bb6fee88126eb85bc21b2bfdc4f` |
| `run_port_reliability_oof_gradient_chain_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `4a8ef412ec7c998f2bc8331e3dab9f92163f58b3766bc41b93efbe4b0458784a` |
| `run_port_reliability_oof_prevention_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `b7a952615316dfaf6028b2541432839f9b6c73094e685487014dd92b1651b99d` |
| `run_port_reliability_oof_prevention_v1.sh` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `86f610430ddde16cb7199c8c3fea29678dfdf317a655cc6dabb23bfdd4fc051e` |
| `run_symeood_c_frozen_readers_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `82a7c577b26d513954bb0c55c4b62bf1a931ce6d40177e4afd22430ef1400314` |
| `run_symeood_mal_pair_v1.py` | 已结束的MAL／S_MAL专项入口或结果复核 | `61c5b387d765f02b97557ca3691f6fff8f8e5d5daca3c15fdea809484964518d` |
| `run_symeood_usg_v1.py` | 已结束的USG专项准备／训练／评价入口 | `6449c3976198ff8907e2d41940139f2a7071324a175b5c88b8520f710d1d4aa5` |
| `summarize_port_size_a20_protection_downstream_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `c5ca276967e871f74f816987b3b8d663bf57ce03be66097fe774dc3d3f3feec8` |
| `summarize_symeood_usg_results_v1.py` | 已结束的USG专项准备／训练／评价入口 | `a914b001f798977c2701ef038be9017c00a4ef3946f2f3ba928e97176df4b88f` |
| `test_port_oof_task_budget_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `8bc2c78ed7291ee178d634b261b824d21639ed10aa1e3cd23d2ad515e87f5675` |
| `test_port_reliability_oof_regression_budget_v1.py` | 已关闭OOF分支的一次性内存／更新诊断及预防入口 | `16c582caabe340240951e99a28316631fd2e6681a3e3248843ce84e86416b7e8` |
| `verify_port_size_proposal_downstream_v1.py` | 已结束专项的独立分析／工程入口或无调用的配套协议；可按Git恢复重放 | `d3192ca1b82939fc8679edde7b626a4c85f9825290dc98e207923b696cc7c322` |
| `verify_symeood_c_frozen_metadata_v1.py` | 已完成的旧跨入口／C诊断／身份准备脚本 | `c823a6da4dd56a9d563160d8727349f694a685aa477e96a7a6f8c2fd12572cd5` |

在项目根目录查看旧内容：

```bash
git show 030395be620d743b64db3bb94bb9ae8d8eb05828:crane_project/tools/analyze_port_geometry_size_axis_balance_v1.py
```

仅在确需恢复且目标路径没有新改动时执行：

```bash
git restore --source=030395be620d743b64db3bb94bb9ae8d8eb05828 -- crane_project/tools/analyze_port_geometry_size_axis_balance_v1.py
```

本次未运行模型、训练、推理、TEST或测距，未同步服务器，未自动提交／推送。实际验证是静态依赖／配置闭包、Git字节及当前冻结来源校验；完整数值测试未在此说明中冒称执行。

## 清理后验证

已完成保留Python语法、JSON解析和静态调用闭包核对；714份保留的原文件与整理前Git字节一致，137项当前链路目标保留，midpoint的127项冻结来源SHA校验通过。使用现有 bundled Python（含NumPy）运行当前midpoint、LVSR共同阶段及固定下游的3组现有unittest，共26项全部通过；这些是接口／身份／方法复用测试，没有模型训练或数据推理。系统Python首次运行因缺NumPy有1项未执行，换用现有环境后已补齐，未安装新依赖。
